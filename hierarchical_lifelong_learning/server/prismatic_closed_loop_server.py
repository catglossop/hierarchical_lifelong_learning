import requests
from io import BytesIO
from PIL import Image
import numpy as np
import urllib.request
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys
from pathlib import Path

import inspect

import jax
import numpy as np
import orbax.checkpoint
import tensorflow as tf
from absl import app, flags
import torch

import wandb
from susie.jax_utils import (
    initialize_compilation_cache,
)
from susie.model import create_sample_fn
from prismatic import load

# jax diffusion stuff
from absl import app as absl_app
from absl import flags
from PIL import Image
import jax
import jax.numpy as jnp

# flask app here
import base64
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

# create rng
rng = jax.random.PRNGKey(0)

from datetime import datetime
import os
from collections import deque
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
from typing import Callable, List, Tuple
from flask import Flask, request, jsonify
import imageio
import jax
import numpy as np
from absl import app, flags
##############################################################################
app = Flask(__name__)


# For gated LMs like Llama-2, make sure to request official access, and generate an access token
hf_token = Path("/home/noam/.cache/huggingface/token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
model_id = "prism-dinosiglip+7b"
vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=torch.bfloat16)

TASK = "Go to the blue garbage bin."

def image_to_base64(image):
    buffer = BytesIO()
    # Convert the image to RGB mode if it's in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

@app.route('/gen_plan', methods=["POST"])
def gen_ll_plan():
    # Receive data 
    print("Got request for plan")
    data = request.get_json()
    img_data = base64.b64decode(data['actions'])
    curr_obs = Image.open(BytesIO(img_data))

    # hl_prompt = data['hl_prompt']
    hl_prompt = TASK
    planning_context = f"""A robot is moving through an indoor environment. The robot is currently executing the task '{hl_prompt}'. 
                           Given the current observation and this task, decompose the high level task into subtasks that can be completed to achieve this task. 
                           where the "GEN_NEW_PLAN" token indicates where the plan is not yet certain. There should be less than 5 subtasks in this plan. If it seems that the high level task has been completed (ie. the object has been reached and is approximately less than 0.5 meters away or the task is done), 
                           set 'task_success' to "yes" in your response. Format your response as a JSON as follows: '"plan":["subtask_1", "subtask_2"...],"task_success":"<"yes" or "no">","reason":"<reasoning>"' where the 'reason' field contains the reasoning for
                           the plan. Return nothing but the response in this form and make sure to use double quotes for the keys and values"""
    start = time.time()
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=planning_context)
    prompt_text = prompt_builder.get_prompt()
    
    vlm_response = vlm.generate(
        curr_obs,
        prompt_text,
        do_sample=True,
        temperature=0.4,
        max_new_tokens=512,
        min_length=1,
    )
    print("VLM response took: ", time.time() - start)
    print(vlm_response)
    response = jsonify(plan=vlm_response)
    return response

@app.route('/verify_action', methods=["POST"])
def verify_action():
    # Receive data 
    data = request.get_json()
    img_data = base64.b64decode(data['actions'])
    curr_obs = Image.open(BytesIO(img_data))
    ll_prompt = data['ll_prompt']

    hl_prompt = TASK
    action_context = f"""A robot is moving through an indoor environment. The robot has been tasked with the high level task '{hl_prompt}' and is executing the subtask {ll_prompt} to complete this task. 
                           We provide an annotated version of the robot's current observation with trajectories it can take projected onto the image in cyan, magenta, yellow, green, blue, and red. 
                           Select the trajectory which will lead the robot to complete the subtask. If none of the trajectories immediately accomplish the task '{hl_prompt}', select the trajectory which will help the robot
                           explore the environment to complete the current subtask. If it seems that the task has been completed (ie. the object has been reached and is approximately less than 0.5 meters away or the task is done), 
                           set 'task_success' to "yes" in your response. Format your response as a JSON as follows: '"trajectory":"<color of the trajectory>","task_success":"<"yes" or "no">","reason":"<reasoning>"'. Return nothing but the response
                           in this form and make sure to use double quotes for the keys and values."""
    start = time.time()
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=action_context)
    prompt_text = prompt_builder.get_prompt()
    
    vlm_response = vlm.generate(
        curr_obs,
        prompt_text,
        do_sample=True,
        temperature=0.4,
        max_new_tokens=512,
        min_length=1,
    )
    print("VLM response took: ", time.time() - start)

    print(vlm_response)
    response = jsonify(traj=vlm_response)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
