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
# Import OpenAI params 
gpt_model = "gpt-4o"
##############################################################################
app = Flask(__name__)


OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID")
client = OpenAI(api_key=OPENAI_KEY,
                    organization = ORGANIZATION_ID)
gpt_model = gpt_model
message_buffer = []
DEBUG = False 
PRIMITIVES = ["Go forward", "Turn left", "Turn right", "Stop"]
TASK = "Go to the kitchen."

def image_to_base64(image):
    buffer = BytesIO()
    # Convert the image to RGB mode if it's in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

@app.route('/gen_hl_instruct', methods=["POST"])
def gen_hl_instruct():
    # Receive data
    data = request.get_json()
    img_data = base64.b64decode(data['curr'])
    curr_obs = Image.open(BytesIO(img_data))

    pass

@app.route('/gen_plan', methods=["POST"])
def gen_ll_plan():
    # Receive data 
    data = request.get_json()
    img_data = base64.b64decode(data['actions'])

    curr_obs = Image.open(BytesIO(img_data))
    curr_obs_64 = image_to_base64(curr_obs)

    # hl_prompt = data['hl_prompt']
    hl_prompt = TASK
    planning_context = f"""A robot is moving through an indoor environment. The robot is currently executing the task '{hl_prompt}'. 
                           We provide an annotated version of the robot's current observation with trajectories it can take projected onto the image in cyan, magenta, yellow, green, blue, and red. 
                           Select the trajectory which will lead the robot to complete the task. If none of the trajectories immediately accomplish the task '{hl_prompt}', select the trajectory which will help the robot
                           explore the environment to find the goal. If it seems that the task has been completed (ie. the object has been reached and is approximately less than 0.5 meters away or the task is done), set 'task_success' to True in your response. Format your response as a JSON as follows: '"trajectory":"<color of the trajectory>","task_success":"<true or false>","reason":"<reasoning>"'. Return nothing but the response
                           in this form and make sure to use double quotes for the keys and values."""
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
    selected_trajectory = ai_response.choices[0].message.content
    print(selected_trajectory)
    response = jsonify(traj=selected_trajectory)
    return response

@app.route('/verify_action', methods=["POST"])
def generate_subgoal():
    # Receive data
    data = request.get_json()
    img_data = base64.b64decode(data['curr'])
    curr_obs = Image.open(BytesIO(img_data))
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
