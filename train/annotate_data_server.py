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

from dlimp.dataset import DLataset
from multinav.deploy.common.trainer_bridge_common import (
    task_data_format,
    make_trainer_config,
)
from multinav.deploy.train.agent import Agent
from multinav.deploy.train.utils import average_dict, average_dicts

from multinav.deploy.train.load_data import (
    dataset_postprocess,
    load_dataset,
    setup_datasets,
    dataset_preprocess,
)

from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

from agentlace.data.rlds_writer import RLDSWriter
import atexit



# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # NO CUDA MEMORY SPENT


FLAGS = flags.FLAGS



def main(_):
    tf.get_logger().setLevel("WARNING")

    data_config: ConfigDict = FLAGS.data_config

    # WITH SAVING 
    data_spec = task_data_format()
    data_dir = "/nfs/nfs2/users/cglossop/lifelong_data"
    existing_folders = [0] + [int(folder.split('.')[-1]) for folder in os.listdir(data_dir)]
    latest_version = max(existing_folders)

    version= f"0.0.{1 + latest_version}"
    datastore_path = f"{data_dir}/{version}"
    os.makedirs(datastore_path)

    writer = RLDSWriter(
        dataset_name="lifelong",
        data_spec = data_spec,
        data_directory = datastore_path,
        version = version,
        max_episodes_per_file = 100,
    )

    atexit.register(writer.close) # so it SAVES on exit

    online_dataset_datastore = EpisodicTFDataStore(
        capacity=10000,
        data_spec= task_data_format(),
        rlds_logger = writer
    )
    print("Datastore set up")

    def request_callback(_type, _payload):
        if _type == "send-stats":
            pass
        elif _type == "get-model-config":
            return model_config # .agent_config.to_dict() # .agent_config get the WHOLE thing
        else:
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

    online_dataset = dataset_preprocess(
        online_dataset_datastore.as_dataset(),
        waypoint_spacing=0.05,
        x_offset= -3,
        angle_scale = 1,
        assign_goal=True,
        end_is_crash=False,
        discount=data_config.discount,
        min_length=2,
        skip_crash = False,
        discrete = False,
        skip_0_x = True,
        truncate_goal = True,
        action_key = "action",
        has_goal = True, 
    )
    online_dataset = dataset_postprocess(online_dataset)

    if train_dataset is None: # online only 
        data_iter = online_dataset
    else: # mix in with existing dataset
        print("Mixing Offline and Online Data")
        data_iter = DLataset.sample_from_datasets(
                [train_dataset, online_dataset],
                weights = [0.5, 0.5])
        # raise NotImplementedError("Offline data not supported")

    data_iter = DLataset.batch(data_iter,
                               model_config.batch_size // num_devices, 
                               drop_remainder = True, 
                               num_parallel_calls = None)
    data_iter = DLataset.batch(data_iter,
                               num_devices, 
                               drop_remainder = True, 
                               num_parallel_calls = None)
    data_iter = DLataset.iterator(data_iter)

    training_data_prefetch = flax.jax_utils.prefetch_to_device(data_iter, 2)

    # ipdb.set_trace() # BREAKPOINT!!! this is how they work


if __name__ == "__main__":
    import os

    config_flags.DEFINE_config_file(
        "model_config",
        os.path.join(os.path.dirname(__file__), "model_config.py:gc_cql"),
        "Configuration for the agent",
    )

    config_flags.DEFINE_config_file(
        "data_config",
        os.path.join(os.path.dirname(__file__), "data_config.py:oppenheimer"),
        "Configuration for the agent",
    )

    flags.DEFINE_string("dataset_name", "gray_local", "Which dataset to train on")

    flags.DEFINE_integer("seed", 42, "Seed for training")

    flags.DEFINE_string("checkpoint_save_dir", None, "Where to store checkpoints")

    flags.DEFINE_string("checkpoint_load_dir", None, "Where to load checkpoints")
    flags.DEFINE_integer("checkpoint_load_step", None, "Which step to load checkpoints")

    # flags.DEFINE_string("data_dir", None, required=True, help="Dataset directory")
    flags.DEFINE_string("wandb_name", None, help="Name of run on W&B")
    flags.DEFINE_string(
        "wandb_dir", "/tmp/wandb", "Where to store temporary W&B data to sync to cloud"
    )

    flags.DEFINE_integer("dataset_update_interval", 100, "Interval between dataset updates")
    flags.DEFINE_integer("wandb_interval", 10, "Interval between calls to wandb.log")
    flags.DEFINE_integer(
        "checkpoint_interval", 500, "Interval between checkpoint saves"
    )

    app.run(main)