"""
Converts data from the BridgeData raw format to TFRecord format.
Consider the following directory structure for the input data:
    bridgedata_raw/
        rss/
            toykitchen2/
                set_table/
                    00/
                        2022-01-01_00-00-00/
                            collection_metadata.json
                            config.json
                            diagnostics.png
                            raw/
                                traj_group0/
                                    traj0/
                                        obs_dict.pkl
                                        policy_out.pkl
                                        agent_data.pkl
                                        images0/
                                            im_0.jpg
                                            im_1.jpg
                                            ...
                                    ...
                                ...
                    01/
                    ...
The --depth parameter controls how much of the data to process at the
--input_path; for example, if --depth=5, then --input_path should be
"bridgedata_raw", and all data will be processed. If --depth=3, then
--input_path should be "bridgedata_raw/rss/toykitchen2", and only data
under "toykitchen2" will be processed.
Can write directly to Google Cloud Storage, but not read from it.
"""
import cv2 as cv
import glob
import logging
import os
import pickle
import random
from datetime import datetime
from functools import partial
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from google.cloud import storage
import tqdm
from absl import app, flags
from tqdm_multiprocess import TqdmMultiProcessPool
from typing import Tuple, Dict, Any, Callable, Sequence, Union
import pandas as pd
import h5py
import dlimp as dl
from dlimp.utils import resize_image, tensor_feature
from hierarchical_lifelong_learning.data.data_utils import (
    make_dataset,
    relabel_primitives, 
    relabel_vlm,
)
FLAGS = flags.FLAGS
flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_string("start_date", None, "start date", required=True)
flags.DEFINE_string("end_date", None, "end date", required=True)
flags.DEFINE_string("primitive", None, "primitive")
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.95, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_integer("shard_size", 200, "Maximum number of trajectories per shard")
IMAGE_SIZE = (256, 256)
POS_THRESHOLD = 0.2 
YAW_THRESHOLD = np.pi/6
CHUNK_SIZE = 10

def tensor_feature(value, key):
    # print(len(value))
    # print(key)
    # if key == "lang":
    #     print(len(value))
    #     print(value)
    # if key == "obs":
    #     print("num of obs: ", len(value))
    # if key =="goal":
    #     print("goal: ", len(value))
    # if key == "gt_lang":
    #     print("gt: ", len(value))

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )

def resize_encode_image(image, size, visualize=True) -> tf.Tensor:
    """Reads, decodes, resizes, and then re-encodes an image."""
    image = tf.io.decode_image(image, expand_animations=False)
    image = resize_image(image, size)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8[p)
    return tf.io.encode_jpeg(image, quality=95)

def flatten_dict(d: Dict[str, Any], sep="/") -> Dict[str, Any]:
    """Given a nested dictionary, flatten it by concatenating keys with sep."""
    flattened = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in flatten_dict(v, sep=sep).items():
                flattened[k + sep + k2] = v2
        else:
            flattened[k] = v
    return flattened

def process_images(images):  # processes images at a trajectory level
    out_images = []
    for image in images:
        out_images.append(resize_encode_image(image, IMAGE_SIZE))
    return out_images

def parse_example(traj):
    os = traj.features.feature["steps/observation/obs"].bytes_list.value
    obs = [process_images([img]) for img in os]

    gt_l = traj.features.feature["steps/observation/gt_lang"].bytes_list.value
    gt_lang = gt_l[0]

    pos = traj.features.feature["steps/observation/position"].float_list.value
    position = [np.array([pos[i], pos[i+1]]) for i in range(0,len(pos)-1,2)]

    ys = traj.features.feature["steps/observation/yaw"].float_list.value
    yaw = [np.array(y) for y in ys]

    goals = traj.features.feature["steps/observation/goal"].bytes_list.value
    goal = process_images([goals[0]])

    out = {}

    out["obs"] = obs
    out["gt_lang"] = gt_lang
    out["position"] = np.array(position)
    out["yaw"] = np.array(yaw)
    out["goal"] = goal

    return out

# create a tfrecord for a group of trajectories
def create_tfrecord(paths, output_path, primitive, tqdm_func, global_tqdm):
    writer = tf.io.TFRecordWriter(output_path)
    for path in paths:
        data = iter(tf.data.TFRecordDataset(path))
        for traj in data:
            traj = traj.numpy()
            example = tf.train.Example()
            example.ParseFromString(traj)
            parsed_example = parse_example(example)
            if primitive:
                # returns only examples of requested primtive
                relabelled_samples = relabel_primitives(parsed_example, primitive, CHUNK_SIZE, YAW_THRESHOLD, POS_THRESHOLD)
            else:
                # use general VLM relabelling 
                relabelled_samples = relabel_vlm(parsed_example)

            for sample in relabelled_samples:
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            k: tensor_feature(v, k) for k, v in flatten_dict(sample).items()
                        }
                    )
                )
                writer.write(example.SerializeToString())

        global_tqdm.update(1)
    writer.close()
    global_tqdm.write(f"Finished {output_path}")

def list_blobs(bucket_name, folder):
    """Lists all the blobs in the bucket."""

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)
    paths = []
    for blob in blobs:
        if blob.name.split("/")[0] == folder: 
            paths.append(blob)
    return paths

def get_lifelong_paths(data_path: str, dates: tuple) -> list:

    # Get the paths to the data we want
    paths = []
    data_folder = "lifelong_datasets"
    folders = list_blobs(data_path, data_folder)
    for folder in folders:
        date = folder.name.split("/")[1]
        if date > dates[0] and date < dates[1]:
            dataset_path = ("/").join(folder.name.split("/")[:-3])
            if folder.name.endswith(".tfrecord", -15, -6):
                new_path = os.path.join("gs://", data_path, folder.name)
                paths.append(new_path)
            
    return paths

def main(_):
    if tf.io.gfile.exists(FLAGS.output_path):
        if FLAGS.overwrite:
            logging.info(f"Deleting {FLAGS.output_path}")
            tf.io.gfile.rmtree(FLAGS.output_path)
        else:
            logging.info(f"{FLAGS.output_path} exists, exiting")
            return
    # each path is a directory that contains dated directories
    paths = get_lifelong_paths(FLAGS.input_path, (FLAGS.start_date, FLAGS.end_date))
    n_train_exs = int(len(paths) * FLAGS.train_proportion)
    train_paths = paths[:n_train_exs]
    val_paths = paths[n_train_exs:]
    random.shuffle(train_paths)
    random.shuffle(val_paths)
    # shard paths
    train_shards = np.array_split(
        train_paths, np.ceil(len(train_paths) / FLAGS.shard_size)
    )
    val_shards = np.array_split(val_paths, np.ceil(len(val_paths) / FLAGS.shard_size))
    # create output paths
    tf.io.gfile.makedirs(os.path.join(FLAGS.output_path, "train"))
    tf.io.gfile.makedirs(os.path.join(FLAGS.output_path, "val"))

    train_output_paths = [
        os.path.join(FLAGS.output_path, "train", f"{i}.tfrecord")
        for i in range(len(train_shards))
    ]
    val_output_paths = [
        os.path.join(FLAGS.output_path, "val", f"{i}.tfrecord")
        for i in range(len(val_shards))
    ]
    # create tasks (see tqdm_multiprocess documenation)
    tasks = [
        (create_tfrecord, (train_shards[i], train_output_paths[i], FLAGS.primitive))
        for i in range(len(train_shards))
    ] + [
        (create_tfrecord, (val_shards[i], val_output_paths[i], FLAGS.primitive))
        for i in range(len(val_shards))
    ]
    # for fn, (a1, a2, a3) in tasks:
    #     fn(a1, a2, a3, None, None)
    # run tasks
    pool = TqdmMultiProcessPool(FLAGS.num_workers)
    with tqdm.tqdm(
        total=len(train_paths) + len(val_paths),
        dynamic_ncols=True,
        position=0,
        desc="Total progress",
    ) as pbar:
        pool.map(pbar, tasks, lambda _: None, lambda _: None)
if __name__ == "__main__":
    app.run(main)
