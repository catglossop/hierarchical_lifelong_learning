import time
from typing import Mapping
from agentlace.data.tf_agents_episode_buffer import EpisodicTFDataStore
from agentlace.trainer import TrainerServer
import wandb
import tqdm
import numpy as np
import tensorflow as tf
import einops
import chex
import sys 
import os
import ipdb
from datetime import datetime
# from ml_collections import config_dict, config_flags, ConfigDict

# from dlimp.dataset import DLataset
from hierarchical_lifelong_learning.train.task_utils import (
    task_data_format,
    make_trainer_config,
)

from hierarchical_lifelong_learning.data.annotate_primitives import (
    dataset_preprocess,
    compute_lang_instruc,
    get_yaw_delta
)

import atexit
import flags



# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # NO CUDA MEMORY SPENT

FLAGS = flags.FLAGS

def main(_):
    tf.get_logger().setLevel("WARNING")

    # WITH SAVING 
    data_spec = task_data_format()
    gcp_bucket = "gs://catg_central2"
    now = datetime.now() 
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    data_dir = "lifelong_data_"

    version= "0.0.0"
    datastore_path = f"{gcp_bucket}/{data_dir}/{version}"
    os.makedirs(datastore_path)
    writer = tf.python_io.TFRecordWriter(datastore_path)

    atexit.register(writer.close) # so it SAVES on exit

    online_dataset_datastore = EpisodicTFDataStore(
        capacity=10000,
        data_spec=task_data_format(),
    )
    print("Datastore set up")

    def request_callback(_type, _payload):
        raise NotImplementedError(f"Unknown request type {_type}")

    train_server = TrainerServer(
        config=make_trainer_config(),
        request_callback=request_callback,
    )
    train_server.register_data_store("online_data", online_dataset_datastore)
    train_server.start(threaded=True)

    samples_to_wait_for = 1000  # usually 1000
    pbar = tqdm.tqdm(total=samples_to_wait_for, desc="Waiting for data")
    while online_dataset_datastore.size < samples_to_wait_for:
        time.sleep(1.0)
        pbar.update(online_dataset_datastore.size - pbar.n)

    processed_dataset = dataset_preprocess(
        online_dataset_datastore.as_dataset(),
        chunk_size=10,
        yaw_threshold=np.pi/2,
        pos_threshold=0.1,
    )

    for data in processed_dataset:
        writer.write(data.SerializeToString())

    # ipdb.set_trace() # BREAKPOINT!!! this is how they work


if __name__ == "__main__":
    import os


    config_flags.DEFINE_config_file(
        "data_config",
        os.path.join(os.path.dirname(__file__), "data_config.py:oppenheimer"),
        "Configuration for the agent",
    )
    flags.DEFINE_string("dataset_name", "gray_local", "Which dataset to train on")
    app.run(main)