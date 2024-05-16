import requests
from io import BytesIO
from PIL import Image
import numpy as np
import urllib.request
import os
from PIL import Image
import matplotlib.pyplot as plt

# jax diffusion stuff
from absl import app as absl_app
from absl import flags
import wandb
from flax.training import checkpoints
from denoising_diffusion_flax.model import EmaTrainState, create_model_def
from denoising_diffusion_flax import scheduling
from denoising_diffusion_flax.sampling import sample_loop
from PIL import Image
import jax
import jax.numpy as jnp
import glob
import shutil
import ml_collections
import time

# flask app here
import base64
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

CHECK_POINT_PATH = "gs://catg_central2/logs/susie-nav_2024.04.08_22.09.33"
WANDB_NAME = "catglossop/susie/susie-nav_2024.04.08_22.09.33"

# create rng
rng = jax.random.PRNGKey(0)

# FLAGS = flags.FLAGS
def image_to_base64(image):
    buffer = BytesIO()
    # Convert the image to RGB mode if it's in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

app = Flask(__name__)
@app.route("/gen_subgoals", methods=["POST"])
def gen_subgoals():
    data = request.json
    image_data = base64.b64decode(data["image"])
    obs_image = Image.open(BytesIO(image_data))
    obs_image.save("obs.png")
    obs_image = np.array(obs_image.resize((128, 128), Image.Resampling.LANCZOS))[..., :3] / 127.5 - 1.0
    obs_images = np.array([obs_image])
    # prepare inputs
    images_repeated = jnp.repeat(
        obs_images, data["num_samples"], axis=0
    )  # (num_images * num_samples, 128, 128, 3)

    # # shard inputs across devices
    # sharding = jax.sharding.PositionalSharding(jax.local_devices())
    # images = jax.device_put(
    #     images, sharding.reshape(sharding.shape[0], *((1,) * (images.ndim - 1)))
    # )

    # sample
    samples = sample_loop(
        rng,
        state,
        images_repeated,
        log_snr_fn=log_snr_fn,
        num_timesteps=data["num_timesteps"],
        w=4.0,
        eta=0.1,
        self_condition=config.ddpm.self_condition,
    )  # (num_images * num_samples, 128, 128, 3)
    samples = jnp.clip(samples * 127.5 + 127.5 + 0.5, 0, 255).astype(jnp.uint8)
    # print("samples.shape", samples.shape)

    # # convert to PIL images
    # print(jax.device_get(samples)[0].shape)
    samples = [sample for sample in jax.device_get(samples)]
    # stack
    samples = np.stack(samples, axis=0).astype(np.uint8)
    # # save images
    # for i, sample in enumerate(samples):
    #     sample.save(f"sample_{i}.png")

    return jsonify({"samples": samples.tolist()})

if __name__ == "__main__":
    # load model here
    api = wandb.Api()
    run = api.run(WANDB_NAME)
    config = ml_collections.ConfigDict(run.config["config"])
    # create model def
    model_def = create_model_def(
        config.model,
    )
    print("Loading weights from checkpoint...", CHECK_POINT_PATH)
    ckpt_dict = checkpoints.restore_checkpoint(CHECK_POINT_PATH, target=None)
    state = EmaTrainState(
        step=0,
        apply_fn=model_def.apply,
        params=None,
        params_ema=ckpt_dict["params_ema"],
        tx=None,
        opt_state=None,
    )

    # parse ddpm params
    log_snr_fn = scheduling.create_log_snr_fn(config.ddpm)
    app.run(host="0.0.0.0", port=5000)

