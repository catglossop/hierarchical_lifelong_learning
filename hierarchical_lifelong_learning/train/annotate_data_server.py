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
from threading import Lock 

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
    data_dir = f"lifelong_datasets/{date_time}/lifelong_data"
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
    lock = Lock()
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

    train_server = TrainerServer(
        config=make_trainer_config(),
        request_callback=request_callback,
    )
    train_server.register_data_store("lifelong_data", online_dataset_datastore)
    train_server.start(threaded=True)

    samples_to_wait_for = 10000  # usually 1000
    pbar = tqdm.tqdm(total=samples_to_wait_for, desc="Waiting for data")
    while True: 
        time.sleep(1.0)
        pbar.update(online_dataset_datastore.size - pbar.n)
        
        #lock.acquire()
        #raw_dataset = online_dataset_datastore.as_dataset().iterator()
        #idx = 0
        #curr_traj_idx = 0
        #len_trajs = 0 
        #for episode in raw_dataset:
        #    traj_idx = episode["_traj_index"][0]
        #    print("IDX IS: ", idx)
        #    idx += 1
        #    print("Traj index: ", traj_idx)
        #    if traj_idx < curr_traj_idx and traj_idx != -1:
        #        break 
        #    if traj_idx == -1: 
        #        len_trajs += episode["_len"][0]
        #        continue
        #    if traj_idx != -1:
        #        curr_traj_idx = traj_idx
        #       processed_episode = relabel_primitives(episode, chunk_size=10, yaw_threshold=np.pi/2, pos_threshold=0.1)
        #        print("One episode became: ", len(processed_episode))
        #        for sample in processed_episode: 
        #            writer(sample)
        #        len_trajs += episode["_len"][0]

        #    if len_trajs >= online_dataset_datastore._num_data_seen: 
        #        break

        #online_dataset_datastore._replay_buffer._replay_buffer.clear()
        #online_dataset_datastore._num_data_seen = 0
        #lock.release()
        #print(online_dataset_datastore.size)
        #ipdb.set_trace()
        


    # ipdb.set_trace() # BREAKPOINT!!! this is how they work


if __name__ == "__main__":
    import os

    app.run(main)
