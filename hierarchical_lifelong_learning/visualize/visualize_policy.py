import base64
import requests
from PIL import Image
from io import BytesIO
import argparse
import matplotlib.pyplot as plt
from hierarchical_lifelong_learning.visualize.action_utils import plot_trajs_and_points_on_image

# ROS
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, UInt8MultiArray, Float32MultiArray
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String, UInt8MultiArray
from nav_msgs.msg import Odometry
from irobot_create_msgs.msg import Dock
from rclpy.action import ActionClient
from irobot_create_msgs.action import Undock
from cv_bridge import CvBridge

from PIL import Image as PILImage
import argparse
import yaml
import numpy as np

# UTILS
from deployment.utils import msg_to_pil, pil_to_msg

from deployment.topic_names import (IMAGE_TOPIC, 
                              SAMPLED_ACTIONS_TOPIC, 
                              ANNOTATED_IMAGE_TOPIC)

from hierarchical_lifelong_learning.visualize.visualize_utils import (BLUE, 
                                                                     GREEN, 
                                                                     RED, 
                                                                     CYAN, 
                                                                     MAGENTA,
                                                                     YELLOW)


# TOPOMAP_IMAGES_DIR = "../topomaps/images"
# MODEL_WEIGHTS_PATH = "../model_weights"
# ROBOT_CONFIG_PATH ="../config/robot.yaml"
# MODEL_CONFIG_PATH = "../config/models.yaml"
# with open(ROBOT_CONFIG_PATH, "r") as f:
#     robot_config = yaml.safe_load(f)
# MAX_V = robot_config["max_v"]
# MAX_W = robot_config["max_w"]
RATE = 4

# IMAGE_TOPIC = "/front/image_raw"
# SAMPLED_ACTIONS_TOPIC = "/sampled_actions"

# GLOBALS
# front_obs_img = None
# reverse_mode = False
# slow_down_gen = False
# sampled_actions = None
# chosen_idx = None
# waypoint = None
# reverse_mode = False
# reverse_obs_img = None
# backtracking = False
# bin_colors = ["red", "yellow", "green"]

class VisualizationNode(Node): 

    def __init__(self):
        super().__init__('low_level_policy')
        self.image_msg = Image()
        self.image = None
        self.image_sub = self.create_subscription(
                                Image, 
                                IMAGE_TOPIC, 
                                self.image_callback, 
                                10)

        self.sampled_actions_msg = Float32MultiArray()
        self.sampled_actions = None
        self.chosen_idx = None
        self.sampled_actions_sub = self.create_subscription(
                                        Float32MultiArray, 
                                        SAMPLED_ACTIONS_TOPIC, 
                                        self.sampled_actions_callback,
                                        1)
        self.annotated_img_pub = self.create_publisher(
                                        Image, 
                                        ANNOTATED_IMAGE_TOPIC, 
                                        1)
        self.timer_period = 1/RATE
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def image_callback(self, msg):
        self.image = msg_to_pil(msg)

    def sampled_actions_callback(self, msg: Float32MultiArray):
        self.chosen_idx = int(msg.data[0])
        sampled_actions_unshaped = np.array(msg.data[1:])
        num_samples = sampled_actions_unshaped.size // (8*2)
        self.sampled_actions = sampled_actions_unshaped.reshape((num_samples, 8, 2))

    def timer_callback(self):
        if (self.image is not None and 
            self.sampled_actions is not None and 
            self.chosen_idx is not None):
            fig, ax = plt.subplots()

            print(f"Chosen idx: {self.chosen_idx}")
            COLORS = [RED, YELLOW, GREEN, CYAN, MAGENTA, BLUE]
            # colors = ["yellow" for _ in range(len(self.sampled_actions))]
            print("len sampled _actions", len(self.sampled_actions))
            COLORS[self.chosen_idx] = BLUE

            # move the chosen idx to the back so we can see it
            indices = list(range(len(self.sampled_actions)))
            indices.remove(self.chosen_idx)
            indices.append(self.chosen_idx)
            self.sampled_actions = self.sampled_actions[indices]

            plot_trajs_and_points_on_image(ax, 
                                        self.image.resize((640, 480), PILImage.Resampling.NEAREST), 
                                        "recon", self.sampled_actions, 
                                        [], COLORS, 
                                        [])
            ax.set_axis_off()
            plt.tight_layout()
            # show image
            img_buf = BytesIO()
            fig.savefig(img_buf, format='jpg')
            im = PILImage.open(img_buf)

            msg = pil_to_msg(im, encoding="passthrough")
            self.annotated_img_pub.publish(msg)
            plt.close(fig)
            print("Published annotated image")

def main():

    rclpy.init()

    viz_node = VisualizationNode()

    rclpy.spin(viz_node)
    viz_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()