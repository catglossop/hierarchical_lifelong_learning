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
CHECK_POINT_PATH = "gs://catg_central2/logs/susie-nav_2024.04.26_23.01.31/200000/state"
WANDB_NAME = "catglossop/susie/jxttu4lu"
PRETRAINED_PATH = "runwayml/stable-diffusion-v1-5:flax"
prompt_w = 5.0
context_w = 5.0
diffusion_num_steps = 50
num_samples = 10

# Import OpenAI params 
gpt_model = "gpt-4-turbo"
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

OPENAI_KEY = os.environ.get("OPENAI_KEY")
ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID")
client = OpenAI(api_key=OPENAI_KEY,
                    organization = ORGANIZATION_ID)
gpt_model = gpt_model

@app.route('/gen_subgoal', methods=["POST"])
def generate_subgoal():
    # Receive data 
    data = request.get_json()
    img_data = base64.b64decode(data['curr'])
    curr_obs = Image.open(BytesIO(img_data))
    curr_obs.save("obs_v2.png")
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
    os.makedirs(folder, exist_ok=True)
    text_file = open(os.path.join(folder, "gpt_comments.txt"), "a+")

    imageio.imwrite(os.path.join(folder, "gen_subgoal_{}.png".format(idx)), gen_subgoal)
    max_samples = 10
    samples_descrip = []
    samples = []
    curr_obs_64 = image_to_base64(curr_obs)
    gen_subgoal_64 = image_to_base64(Image.fromarray(gen_subgoal))
    while not gpt_approved and idx < max_samples:
        print("Diffusion sample: ", idx)
        ai_response = client.chat.completions.create(
            model=gpt_model,
            messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"A robot is trying to perform the high level task {hl_prompt}. It is currently executing the low level task {ll_prompt}. The first image is the robot's current observation and the second image is the the goal image for the low level prompt. Is this goal image consistent with the current observation, high level task, and low level task? Respond in the form 'YES: [insert an explaination of why this subgoal is good]' if yes and 'NO: [insert an explaination of why the subgoal is bad]' if no."},
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
            ],
            max_tokens=300,
        )
        # process the current response for positive or negative
        curr_response = ai_response.choices[0].message.content
        print("Response: ", curr_response)
        text_file.write(curr_response)
        samples_descrip.append(f"{idx}: {curr_response}")
        samples.append(gen_subgoal_64)
        if ai_response.choices[0].message.content.split(":")[0] == "YES":
            gpt_approved = True
            imageio.imwrite(os.path.join(folder, "gen_subgoal_chosen.png"), gen_subgoal)
            response = jsonify(goal=gen_subgoal_64)
            return response
        idx += 1
        gen_subgoal = diffusion_sample(curr_obs_np, ll_prompt)
        imageio.imwrite(os.path.join(folder, f"gen_subgoal_{idx}.png"), gen_subgoal)

    samples_descrip_processed = (" ").join(samples_descrip)
    print(samples_descrip_processed)
    ai_response = client.chat.completions.create(
        model=gpt_model,
        messages=[
        {
        "role": "user",
        "content": [

            {"type": "text", "text": f"A robot is trying to perform the high level task {hl_prompt}. It is currently executing the low level task {ll_prompt}. {max_samples} subgoals were generated for this task and all of them were deemed not good enough. Here are the descriptions of why each of the subgoals were wrong: {samples_descrip_processed}. Respond with the number corresponding to the best option of these subgoals and the number only. If they are all equally bad, return a random number between 0 and {max_samples - 1}. Respond with the number only."},
            ],
        }
        ],
        max_tokens=300,
    )
    text_file.write(ai_response.choices[0].message.content)
    print("Selected subgoal is: ", int(ai_response.choices[0].message.content))
    gen_subgoal_selected = samples[int(ai_response.choices[0].message.content)]
    imageio.imwrite(os.path.join(folder, "gen_subgoal_chosen.png"), gen_subgoal)
    response = jsonify(goal=gen_subgoal_selected)
    text_file.close()
    return response


def image_to_base64(image):
    buffer = BytesIO()
    # Convert the image to RGB mode if it's in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)