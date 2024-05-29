import requests
from io import BytesIO
from PIL import Image
import numpy as np
import urllib.request
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys

import inspect

import jax
import numpy as np
import orbax.checkpoint
import tensorflow as tf
from absl import app, flags

import wandb
from susie.jax_utils import (
    initialize_compilation_cache,
)
from susie.model import create_sample_fn

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
from openai import OpenAI
##############################################################################

# Diffusion model params 
#CHECK_POINT_PATH = "gs://catg_central2/logs/susie-nav_2024.04.26_23.01.31/200000/state"
#WANDB_NAME = "catglossop/susie/jxttu4lu"
CHECK_POINT_PATH = "gs://catg_central2/logs/susie-nav_2024.05.02_09.55.29/100000/state"
WANDB_NAME = "catglossop/susie/jfwhcabr"
PRETRAINED_PATH = "runwayml/stable-diffusion-v1-5:flax"

prompt_w = 4.0
context_w = 6.0
diffusion_num_steps = 50
num_samples = 10

# Import OpenAI params 
gpt_model = "gpt-4o"
##############################################################################
app = Flask(__name__)

diffusion_sample = create_sample_fn(
        CHECK_POINT_PATH,
        WANDB_NAME,
        diffusion_num_steps,
        prompt_w,
        context_w,
        0.0,
        PRETRAINED_PATH,
    )

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID")
client = OpenAI(api_key=OPENAI_KEY,
                    organization = ORGANIZATION_ID)
gpt_model = gpt_model
message_buffer = []
DEBUG = False 

def image_to_base64(image):
    buffer = BytesIO()
    # Convert the image to RGB mode if it's in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

## Load in the context images
context_image_folder = "/nfs/nfs2/users/cglossop/context_images"
context_images = {}
for img in os.listdir(context_image_folder):
    context_images[img.split(".")[0]] = image_to_base64(Image.open(os.path.join(context_image_folder, img)))
    
PRIMITIVES = ["Go forward", "Turn left", "Turn right", "Stop"]
TASKS = ["Go down the hallway", "Go to the chair", "Go to the kitchen","Go to the door", "Follow the person"]
initial_context = f"""A robot is moving through an indoor environment. It is being given language tasks which include the primitive actions {(", ").join(PRIMITIVES)}
                    and the higher level tasks {(", ").join(TASKS)}. The robot has a model that can generate image subgoals conditioned on a language instruction. 
                    We provide examples of good observation and generated subgoal pairs. 
                    The first image is the observation and the second is corresponding generated subgoal image for the primitive {PRIMITIVES[0]}. 
                    The third image is the observation and the fourth is corresponding generated subgoal image for the primitive {PRIMITIVES[1]}.
                    The fifth image is the observation and the sixth is corresponding generated subgoal image for the primitive {PRIMITIVES[2]}. 
                    The seventh image is the observation and the eighth is corresponding generated subgoal image for the primitive {PRIMITIVES[3]}. 
                    There does not need to be a hallway for the robot to go down in the observation. The robot can be in any location in the environment.""" 
initial_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": initial_context},
        {
            "type": "image_url",
            "image_url": {"url":"data:image/jpeg;base64,{}".format(context_images["go_forward_obs"])},
        },
        {
            "type": "image_url",
            "image_url": {"url":"data:image/jpeg;base64,{}".format(context_images["go_forward_goal"])},
        },
        {
            "type": "image_url",
            "image_url": {"url":"data:image/jpeg;base64,{}".format(context_images["turn_left_obs"])},
        },
        {
            "type": "image_url",
            "image_url": {"url":"data:image/jpeg;base64,{}".format(context_images["turn_left_goal"])},
        },
        {
            "type": "image_url",
            "image_url": {"url":"data:image/jpeg;base64,{}".format(context_images["turn_right_obs"])},
        },
        {
            "type": "image_url",
            "image_url": {"url":"data:image/jpeg;base64,{}".format(context_images["turn_right_goal"])},
        },
        {
            "type": "image_url",
            "image_url": {"url":"data:image/jpeg;base64,{}".format(context_images["stop_obs"])},
        },
        {
            "type": "image_url",
            "image_url": {"url":"data:image/jpeg;base64,{}".format(context_images["stop_goal"])},
        },
        ],
    }
message_buffer.append(initial_message)
ai_response = client.chat.completions.create(
    model=gpt_model,
    messages=[
        initial_message
    ],
    max_tokens=300,
)
message_buffer.append(ai_response.choices[0].message)
@app.route('/gen_subgoal', methods=["POST"])
def generate_subgoal():
    # Receive data
    global message_buffer
    data = request.get_json()
    img_data = base64.b64decode(data['curr'])
    curr_obs = Image.open(BytesIO(img_data))
    hl_prompt = data['hl_prompt']
    ll_prompt = data['ll_prompt']
    print("High level language instruction: ", hl_prompt)
    print("Currently executing: ", ll_prompt)
    ll_prompt_mod = ("").join([s[0] for s in ll_prompt.split()])
    # Pass image to GPT
    gpt_approved = False
    idx = 0
    curr_obs_np = np.array(curr_obs.resize((128, 128), Image.Resampling.LANCZOS))[...,:3]
    print("Generating subgoal...")
    gen_subgoal = diffusion_sample(curr_obs_np, ll_prompt)
    time_key = datetime.now().strftime("%Y-%d-%m_%H-%M-%S")
    folder = f"gen_subgoals_{ll_prompt_mod}_{time_key}"
    if DEBUG:
        os.makedirs(folder, exist_ok=True)
        text_file = open(os.path.join(folder, "gpt_comments.txt"), "a+")

    if DEBUG:
        imageio.imwrite(os.path.join(folder, "gen_subgoal_{}.png".format(idx)), gen_subgoal)
    samples_descrip = []
    samples = []
    curr_obs_64 = image_to_base64(curr_obs)
    gen_subgoal_64 = image_to_base64(Image.fromarray(gen_subgoal))
    while not gpt_approved and idx < num_samples:
        print("Diffusion sample: ", idx)
        current_context = f"""The ID of this message is {idx}. A robot is trying to perform the high level task {hl_prompt} (ignore if None). It is currently executing the low level task {ll_prompt}. 
                            The first image is the robot's current observation and the second image is the the goal image for the low level prompt. 
                            Is this goal image consistent with the current observation, high level task, and low level task? 
                            Respond in the form 'YES: [insert an explaination of why this subgoal is good]' if yes and 
                            'NO: [insert an explaination of why the subgoal is bad]' if no."""
        current_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": current_context},
                {
                    "type": "image_url",
                    "image_url": {"url":"data:image/jpeg;base64,{}".format(curr_obs_64)},
                },
                {
                    "type": "image_url",
                    "image_url": {"url":"data:image/jpeg;base64,{}".format(gen_subgoal_64)},
                },
                ],
            }
        message_buffer.append(current_message)

        ai_response = client.chat.completions.create(
            model=gpt_model,
            messages=message_buffer,
            max_tokens=300,
        )
        message_buffer.append(ai_response.choices[0].message)

        # process the current response for positive or negative
        curr_response = ai_response.choices[0].message.content
        print("Response: ", curr_response)
        if DEBUG:
            text_file.write(curr_response)
        samples_descrip.append(f"{idx}: {curr_response}")
        samples.append(gen_subgoal_64)
        if ai_response.choices[0].message.content.split(":")[0] == "YES":
            gpt_approved = True
            if DEBUG:
                imageio.imwrite(os.path.join(folder, "gen_subgoal_chosen.png"), gen_subgoal)
            response = jsonify(goal=gen_subgoal_64, succeeded=True)
            message_buffer = [initial_message]
            return response
        idx += 1
        gen_subgoal = diffusion_sample(curr_obs_np, ll_prompt)
        if DEBUG:
            imageio.imwrite(os.path.join(folder, f"gen_subgoal_{idx}.png"), gen_subgoal)

    # Reset the context buffer
    samples_descrip_processed = (" ").join(samples_descrip)
    print(samples_descrip_processed)
    fallback_context = f"""A robot is trying to perform the high level task {hl_prompt} (ignore if None). It is currently executing the low level task {ll_prompt}. 
                        {num_samples} subgoals were generated for this task and all of them were deemed not good enough. 
                        Choose the best option from the previous examples and return the ID of the best option. The response must only contain the ID of the best option."""
    fallback_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": fallback_context},
            ],
        }
    message_buffer.append(fallback_message)
    ai_response = client.chat.completions.create(
        model=gpt_model,
        messages=message_buffer,
        max_tokens=300,
    )
    
    if DEBUG:
        text_file.write(ai_response.choices[0].message.content)
    print("Selected subgoal is: ", int(ai_response.choices[0].message.content))
    gen_subgoal_selected = samples[int(ai_response.choices[0].message.content)]
    if DEBUG:
        imageio.imwrite(os.path.join(folder, "gen_subgoal_chosen.png"), gen_subgoal)
    response = jsonify(goal=gen_subgoal_selected, succeeded=False)
    if DEBUG:
        text_file.close()
    message_buffer = [initial_message]
    return response

@app.route('/gen_plan', methods=["POST"])
def generate_plan():
    # Receive data 
    data = request.get_json()
    img_data = base64.b64decode(data['curr'])
    curr_obs = Image.open(BytesIO(img_data))
    curr_obs_64 = image_to_base64(curr_obs)

    # Pass image to GPT
    planning_context = f"""A robot is moving through an indoor environment. The provided image is the robot's current observation. 
                           Ultimately, we want the robot to perform the high level tasks {(", ").join(TASKS)}. Given the current observation, 
                           generate a plan in the form of a list of actions the robot should take using only the low level tasks in this list: {(", ").join(PRIMITIVES)} which are executed at with a max angular velocity of 0.2 rad/s and linear velocity of 0.1 m/s over 15 seconds that aims to accomplish the high level task "Go to the door.". 
                           If it seems that none of the high level tasks can be immediately accomplished, generate a reasonable plan as a list of low level tasks that explore the environment to find the high level tasks.
                           Format the list as follows '[insert action], [insert action], [insert action], ...'. If a high level task is being executed, append the high level task to the end of the list. Otherwise, append 'None'. Return nothing but the plan with no additional words."""
    planning_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": planning_context},
        {
            "type": "image_url",
            "image_url": {"url":"data:image/jpeg;base64,{}".format(curr_obs_64)},
        },
        ],
        }
    ai_response = client.chat.completions.create(
            model=gpt_model,
            messages=[planning_message],
            max_tokens=300,
    )
    vlm_plan = ai_response.choices[0].message.content
    print(vlm_plan)
    response = jsonify(plan=vlm_plan)
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
