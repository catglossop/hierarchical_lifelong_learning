import base64
import requests
from io import BytesIO

# ROS
import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, UInt8MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from PIL import Image as PILImage
import argparse
import yaml
import numpy as np
import threading
import cv2
from cv_bridge import CvBridge
from flask import Flask, jsonify, request

# UTILS
from deployment.utils import msg_to_pil, to_numpy, transform_images, load_model
import imageio

class SubGoalClient(Node): 

    def __init__(self, 
                tpu_ip):
        super().__init__("subgoal_client")

        self.SERVER_ADDRESS = f"http://{tpu_ip}:5001/gen_subgoal"
        self.irobot_qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT, 
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
                depth=1
        )
        self.subgoals_msg = None 
        self.subgoal = None 
        self.subgoals_pub = self.create_publisher(
            Image,
            "/hierarchical_learning/subgoal",
            self.irobot_qos_profile)
        self.obs = None
        self.obs_sub = self.create_subscription(
            Image, 
            "/front/image_raw",
            self.obs_callback,
            self.irobot_qos_profile
        )
        self.act_status = True
        self.act_sub = self.create_subscription(
            Bool, 
            "/hierarchical_learning/action_status",
            self.act_callback,
            self.irobot_qos_profile
        )
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.bridge = CvBridge()

    def msg_to_pil(self, msg: PILImage) -> PILImage.Image:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
        pil_image = PILImage.fromarray(img)
        return pil_image
    
    def obs_callback(self, obs_msg):
        print("In obs callback")
        self.obs = self.msg_to_pil(obs_msg)
    
    def act_callback(self, act_msg): 

        self.act_status = act_msg.data

    def image_to_base64(self, image):
        buffer = BytesIO()
        # Convert the image to RGB mode if it's in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return img_str

    def send_image_to_server(self, image: PILImage.Image) -> dict:
        image_base64 = self.image_to_base64(image)
        data = {
            'curr': image_base64,
            'hl_prompt': 'Turn left',
            'll_prompt': 'Turn left',
        }
        response = requests.post(self.SERVER_ADDRESS, json=data, timeout=99999999)
        data = response.json()
        img_data = base64.b64decode(data['goal'])
        subgoal = PILImage.open(BytesIO(img_data))
        subgoal = np.array(subgoal)
        return subgoal
    
    def timer_callback(self):
        if self.obs is not None and self.act_status:
            print("Valid observation available...")
            imageio.imwrite(f"/home/create/curr_obs_go_forward.png", self.obs)
            self.subgoal = self.send_image_to_server(self.obs)
            self.subgoal = np.array(self.subgoal).astype(np.uint8)
            self.subgoals_msg = self.bridge.cv2_to_imgmsg(self.subgoal, "passthrough")
            self.subgoals_pub.publish(self.subgoals_msg)
        elif self.subgoal is not None: 
            self.subgoals_pub.publish(self.subgoals_msg)


def main(args: argparse.Namespace):
    rclpy.init()
    subgoal_client = SubGoalClient(args.ip)
    rclpy.spin(subgoal_client)
    subgoal_client.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='subgoal_client',
                    description='Runs the client for getting diffusion subgoals')
    parser.add_argument('-i', '--ip', type=str, default="http://173.255.124.92:5001/gen_subgoals")  # on/off flag
    args = parser.parse_args()
    main(args)