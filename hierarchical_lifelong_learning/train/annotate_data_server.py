from functools import partial
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
from agentlace.data.rlds_writer import RLDSWriter
# from ml_collections import config_dict, config_flags, ConfigDict

# from dlimp.dataset import DLataset
from hierarchical_lifelong_learning.train.task_utils import (
    task_data_format,
    rlds_data_format,
    make_trainer_config,
)

from hierarchical_lifelong_learning.data.data_utils import (
    relabel_primitives,
    relabel_vlm,
    compute_lang_instruc,
    get_yaw_delta
)

import atexit
from absl import app, flags, logging as absl_logging

def main(_):
    tf.get_logger().setLevel("WARNING")

    # WITH SAVING 
    data_spec = rlds_data_format()
    gcp_bucket = "gs://catg_central2"
    now = datetime.now() 
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    data_dir = f"lifelong_data_{date_time}"
    name = "lifelong_data"
    version= "0.0.0"
    datastore_path = f"{gcp_bucket}/{data_dir}/{version}"
    #writer = tf.io.TFRecordWriter(datastore_path)
    writer = RLDSWriter(
            dataset_name=name,
            data_spec = data_spec,
            data_directory = datastore_path,
            version=version, 
            max_episodes_per_file=100,
    )
    atexit.register(writer.close) # so it SAVES on exit

    online_dataset_datastore = EpisodicTFDataStore(
        capacity=10000,
        data_spec=rlds_data_format(),
        rlds_logger = writer
    )
    print("Datastore set up")

    def request_callback(_type, _payload):
        raise NotImplementedError(f"Unknown request type {_type}")
    train_config = make_trainer_config()
    print(train_config.port_number)
    print(train_config)
    train_server = TrainerServer(
        config=make_trainer_config(),
        request_callback=request_callback,
    )
    train_server.register_data_store("lifelong_data", online_dataset_datastore)
    train_server.start(threaded=True)

    samples_to_wait_for = 40  # usually 1000
    pbar = tqdm.tqdm(total=samples_to_wait_for, desc="Waiting for data")
    while online_dataset_datastore.size < samples_to_wait_for:
        time.sleep(1.0)
        pbar.update(online_dataset_datastore.size - pbar.n)
        print(online_dataset_datastore._num_data_seen)

    # load dataset 
    for file in tf.io.gfile.listdir(datastore_path):
        if not file.startswith(name):
            continue
        print("Dataset found! Starting processing...")
        dataset = tfds.load(name,
            data_dir = datastore_path,
        )

        processed_dataset = relabel_primitives(
            online_dataset_datastore.as_dataset(),
            chunk_size=10,
            yaw_threshold=np.pi/2,
            pos_threshold=0.1,
        )
        new_version = "0.0.1"
        modified_path = f"{gcp_bucket}/{data_dir}/{new_version}"
        processed_dataset.save(modified_path)

    # ipdb.set_trace() # BREAKPOINT!!! this is how they work


if __name__ == "__main__":
    import os

    app.run(main)
