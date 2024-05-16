import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml
import threading

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from deployment.utils import msg_to_pil, to_numpy, transform_images, load_model

import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time

# UTILS
from deployment.topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC, 
                        REACHED_GOAL_TOPIC)


# CONSTANTS
TOPOMAP_IMAGES_DIR = "topomaps/images"
ROBOT_CONFIG_PATH ="../../deployment/config/robot.yaml"
MODEL_CONFIG_PATH = "../../deployment/config/models.yaml"
DATA_CONFIG = "../../deployment/config/data_config.yaml"



class LowLevelPolicy(Node): 

    def __init__(self, 
                args
                ):
        super().__init__('low_level_policy')
        self.args = args
        self.context_queue = []
        self.context_size = None

        # Load the model 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.load_model_from_config(MODEL_CONFIG_PATH)

        # Load the config
        self.load_config(ROBOT_CONFIG_PATH)

        # Load the topomap
        self.load_topomap(TOPOMAP_IMAGES_DIR)

        # Load data config
        self.load_data_config()
        
        # SUBSCRIBERS  
        self.image_msg = Image()
        self.image_sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            1)
        
        # PUBLISHERS
        self.reached_goal = False
        self.reached_goal_msg = Bool()
        self.reached_goal_pub = self.create_publisher(
            Bool, 
            REACHED_GOAL_TOPIC, 
            1)
        self.sampled_actions_msg = Float32MultiArray()
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, 
            SAMPLED_ACTIONS_TOPIC, 
            1)
        self.waypoint_msg = Float32MultiArray()
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, 
            WAYPOINT_TOPIC, 
            1)  
        
        # TIMERS
        self.timer_period = 1/self.RATE  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
    
    # Utils
    def unnormalize_data(self, ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data

    def get_delta(self, actions):
        # append zeros to first action
        ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
        delta = ex_actions[:,1:] - ex_actions[:,:-1]
        return delta

    def get_action(self):
        # diffusion_output: (B, 2*T+1, 1)
        # return: (B, T-1)
        ndeltas = self.naction
        ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
        ndeltas = to_numpy(ndeltas)
        ndeltas = self.unnormalize_data(ndeltas, self.ACTION_STATS)
        actions = np.cumsum(ndeltas, axis=1)
        return torch.from_numpy(actions).to(self.device)

    def load_config(self, robot_config_path):
        with open(robot_config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        self.MAX_V = robot_config["max_v"]
        self.MAX_W = robot_config["max_w"]
        self.VEL_TOPIC = "/task_vel"
        self.DT = 1/robot_config["frame_rate"]
        self.RATE = robot_config["frame_rate"]
        self.EPS = 1e-8
        self.WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
        self.FLIP_ANG_VEL = np.pi/4
    
    def load_model_from_config(self, model_paths_config):
        # Load configs
        with open(model_paths_config, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths["nomad"]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"] 

        # Load model weights
        self.ckpth_path = model_paths["nomad"]["ckpt_path"]
        if os.path.exists(self.ckpth_path):
            print(f"Loading model from {self.ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {self.ckpth_path}")
        self.model = load_model(
            self.ckpth_path,
            self.model_params,
            self.device,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
    def load_topomap(self, topomap_images_dir):
        # Load topomap
        topomap_filenames = sorted(os.listdir(os.path.join(
            topomap_images_dir, self.args.dir)), key=lambda x: int(x.split(".")[0]))
        topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{self.args.dir}"
        self.num_nodes = len(os.listdir(topomap_dir))
        self.topomap = []
        for i in range(self.num_nodes):
            image_path = os.path.join(topomap_dir, topomap_filenames[i])
            self.topomap.append(PILImage.open(image_path))
        
        self.closest_node = 0
        assert -1 <= self.args.goal_node < len(self.topomap), "Invalid goal index"
        if self.args.goal_node == -1:
            self.goal_node = len(self.topomap) - 1
        else:
            self.goal_node = self.args.goal_node
        self.reached_goal = False
    
    def load_data_config(self):
        # LOAD DATA CONFIG
        with open(os.path.join(os.path.dirname(__file__), DATA_CONFIG), "r") as f:
            data_config = yaml.safe_load(f)
        # POPULATE ACTION STATS
        self.ACTION_STATS = {}
        for key in data_config['action_stats']:
            self.ACTION_STATS[key] = np.array(data_config['action_stats'][key])

    def image_callback(self, msg):
        self.image_msg = msg_to_pil(msg)
        img = self.image_msg.save("test_image.jpg")
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(self.image_msg)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(self.image_msg)
    
    def process_images(self):
        self.obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
        self.obs_images = torch.split(self.obs_images, 3, dim=1)
        self.obs_images = torch.cat(self.obs_images, dim=1) 
        self.obs_images = self.obs_images.to(self.device)
        self.mask = torch.zeros(1).long().to(self.device)  
    
    def infer_actions(self):
        # Get early fusion obs goal for conditioning
        self.obsgoal_cond = self.model('vision_encoder', 
                                        obs_img=self.obs_images.repeat(len(self.goal_image), 1, 1, 1), 
                                        goal_img=self.goal_image, 
                                        input_goal_mask=self.mask.repeat(len(self.goal_image)))
        # Predict distances
        self.dists = self.model("dist_pred_net", obsgoal_cond=self.obsgoal_cond)
        self.dists = to_numpy(self.dists.flatten())
        print("DISTANCES: ", self.dists)
        self.min_idx = np.argmin(self.dists)
        self.closest_node = self.min_idx + self.start
        print("closest node:", self.closest_node)
        self.sg_idx = min(self.min_idx + int(self.dists[self.min_idx] < self.args.close_threshold), len(self.obsgoal_cond) - 1)
        self.obs_cond = self.obsgoal_cond[self.sg_idx].unsqueeze(0)

        # infer action
        with torch.no_grad():
            # encoder vision features
            if len(self.obs_cond.shape) == 2:
                self.obs_cond = self.obs_cond.repeat(self.args.num_samples, 1)
            else:
                self.obs_cond = self.obs_cond.repeat(self.args.num_samples, 1, 1)
            
            # initialize action from Gaussian noise
            self.noisy_action = torch.randn(
                (self.args.num_samples, self.model_params["len_traj_pred"], 2), device=self.device)
            self.naction = self.noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            self.start_time = time.time()
            for k in self.noise_scheduler.timesteps[:]:
                # predict noise
                self.noise_pred = self.model(
                    'noise_pred_net',
                    sample=self.naction,
                    timestep=k,
                    global_cond=self.obs_cond
                )
                # inverse diffusion step (remove noise)
                self.naction = self.noise_scheduler.step(
                    model_output=self.noise_pred,
                    timestep=k,
                    sample=self.naction
                ).prev_sample
            print("time elapsed:", time.time() - self.start_time)
            print("DEVICE: " , self.device)

        self.naction = to_numpy(self.get_action())
        self.sampled_actions_msg = Float32MultiArray()
        self.sampled_actions_msg.data = np.concatenate((np.array([0]), self.naction.flatten())).tolist()
        print("published sampled actions")
        self.sampled_actions_pub.publish(self.sampled_actions_msg)
        self.naction = self.naction[0] 
        self.chosen_waypoint = self.naction[self.args.waypoint] 

    def timer_callback(self):

        self.chosen_waypoint = np.zeros(4, dtype=np.float32)
        if len(self.context_queue) > self.model_params["context_size"]:

            # Process observations
            self.process_images()

            # Get goal image
            self.start = max(self.closest_node - self.args.radius, 0)
            self.end = min(self.closest_node + self.args.radius + 1, self.goal_node)
            self.goal_image = [transform_images(g_img, self.model_params["image_size"], center_crop=False).to(self.device) for g_img in self.topomap[self.start:self.end + 1]]
            self.goal_image = torch.concat(self.goal_image, dim=0)

            # Use policy to get actions
            self.infer_actions()

        # Normalize and publish waypoint
        if self.model_params["normalize"]:
            self.chosen_waypoint[:2] *= (self.MAX_V / self.RATE)  

        self.waypoint_msg.data = self.chosen_waypoint.tolist()
        self.waypoint_pub.publish(self.waypoint_msg)

        # Check if goal reached
        self.reached_goal = bool(self.closest_node == self.goal_node)
        print(self.reached_goal, type(self.reached_goal))
        self.reached_goal_msg.data = self.reached_goal
        self.reached_goal_pub.publish(self.reached_goal_msg)
        if self.reached_goal:
            print("Reached goal! Stopping...")


def main(args):
    rclpy.init()
    # node = rclpy.create_node('low_level_policy')
    # thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    # thread.start()
    low_level_policy = LowLevelPolicy(args)

    rclpy.spin(low_level_policy)
    low_level_policy.destroy_node()
    rclpy.shutdown()

    # rate = node.create_rate(low_level_policy.RATE)
    
    print("Registered with master node. Waiting for image observations...")
    print(f"Using {low_level_policy.device}")

    # try:
    #     while rclpy.ok():
    #         low_level_policy.chosen_waypoint = np.zeros(4, dtype=np.float32)
    #         if len(low_level_policy.context_queue) > low_level_policy.model_params["context_size"]:

    #             # Process observations
    #             low_level_policy.process_images()

    #             # Get goal image
    #             low_level_policy.start = max(low_level_policy.closest_node - low_level_policy.args.radius, 0)
    #             low_level_policy.end = min(low_level_policy.closest_node + low_level_policy.args.radius + 1, low_level_policy.goal_node)
    #             low_level_policy.goal_image = [transform_images(g_img, low_level_policy.model_params["image_size"], center_crop=False).to(low_level_policy.device) for g_img in low_level_policy.topomap[low_level_policy.start:low_level_policy.end + 1]]
    #             low_level_policy.goal_image = torch.concat(low_level_policy.goal_image, dim=0)

    #             # Use policy to get actions
    #             low_level_policy.infer_actions()

    #         # Normalize and publish waypoint
    #         if low_level_policy.model_params["normalize"]:
    #             low_level_policy.chosen_waypoint[:2] *= (low_level_policy.MAX_V / low_level_policy.RATE)  

    #         low_level_policy.waypoint_msg.data = low_level_policy.chosen_waypoint.tolist()
    #         low_level_policy.waypoint_pub.publish(low_level_policy.waypoint_msg)

    #         # Check if goal reached
    #         low_level_policy.reached_goal = bool(low_level_policy.closest_node == low_level_policy.goal_node)
    #         print(low_level_policy.reached_goal, type(low_level_policy.reached_goal))
    #         low_level_policy.reached_goal_msg.data = low_level_policy.reached_goal
    #         low_level_policy.reached_goal_pub.publish(low_level_policy.reached_goal_msg)
    #         if low_level_policy.reached_goal:
    #             print("Reached goal! Stopping...")
    #         rate.sleep()
    # except KeyboardInterrupt:
    #     pass

    # rclpy.shutdown()
    # thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    main(args)


