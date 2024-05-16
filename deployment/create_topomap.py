import argparse
import os
from deployment.utils import msg_to_pil 
import time

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy

IMAGE_TOPIC = "/front/image_raw"
TOPOMAP_IMAGES_DIR = "topomaps/images"
obs_img = None


class CreateTopomap(Node): 

    def __init__(self, args):
        super().__init__('create_topomap')
        self.args = args
        self.obs_img = None
        self.obs_sub = self.create_subscription(
            Image, 
            IMAGE_TOPIC, 
            self.obs_callback, 
            1
        )
        self.joy_msg = Joy()
        self.joy_sub = self.create_subscription(
            Joy, 
            "joy", 
            self.joy_callback, 
            1
        )
        self.subgoal_msg = Image()
        self.subgoal_pub = self.create_publisher(
            Image, 
            "/subgoals", 
            1
        )
        self.timer_period = self.args.dt
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, self.args.dir)
        if not os.path.isdir(self.topomap_name_dir):
            os.makedirs(self.topomap_name_dir)
        else:
            print(f"{self.topomap_name_dir} already exists. Removing previous images...")
            self.remove_files_in_dir(self.topomap_name_dir)
            
        assert self.args.dt > 0, "dt must be positive"
        print("Registered with master node. Waiting for images...")
        self.i = 0
        self.start_time = float("inf")

    def remove_files_in_dir(self, dir_path: str):
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

    def obs_callback(self, msg: Image):
        self.obs_img = msg_to_pil(msg)
    
    def joy_callback(self, msg: Joy):
        if msg.buttons[0]:
            self.destroy_node()
            rclpy.shutdown()


    def timer_callback(self): 
        if self.obs_img is not None: 
            self.obs_img.save(os.path.join(self.topomap_name_dir, f"{self.i}.png"))
            print("published image", self.i)
            self.i += 1
            self.start_time = time.time()
            self.obs_img = None
        if time.time() - self.start_time > 2 * self.args.dt:
            print(f"Topic {IMAGE_TOPIC} not publishing anymore. Shutting down...")
            self.destroy_node()
            rclpy.shutdown()

def main(args: argparse.Namespace):

    rclpy.init()

    create_topomap_node = CreateTopomap(args)

    rclpy.spin(create_topomap_node)
    create_topomap_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Code to generate topomaps from the {IMAGE_TOPIC} topic"
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topological map images in ../topomaps/images directory (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=4.,
        type=float,
        help=f"time between images sampled from the {IMAGE_TOPIC} topic (default: 3.0)",
    )
    args = parser.parse_args()

    main(args)
