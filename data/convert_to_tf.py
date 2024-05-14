"""
Converts data from the BridgeData raw format to TFRecord format.

Consider the following directory structure for the input data:

    sacson_raw/
        month-day-year-location-run/
            0.jpg
            ...
            n.jpg
            traj_data.pkl
        

The --depth parameter controls how much of the data to process at the
--input_path; for example, if --depth=1, then --input_path should be
"sacson", and all data will be processed.

Can write directly to Google Cloud Storage, but not read from it.
"""

import glob
import logging
import os
import sys
import pickle
import random
import yaml
from datetime import datetime
from functools import partial
from multiprocessing import Pool
import torch
from typing import Tuple, Dict, Any, Callable, Sequence, Union

import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from absl.flags import FLAGS
from tqdm_multiprocess import TqdmMultiProcessPool

from torch.utils.data import DataLoader, ConcatDataset, Subset

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    5,
    "Number of directories deep to traverse to the dated directory. Looks for"
    "{input_path}/dir_1/dir_2/.../dir_{depth-1}/2022-01-01_00-00-00/...",
)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.8, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_integer("shard_size", 200, "Maximum number of trajectories per shard")

# INDOOR_DATASETS = ["sacson", "cory_hall", "go_stanford_cropped", "scand"]
INDOOR_DATASETS = ["sacson"]
# INDOOR_DATASETS = ["sacson", "cory_hall", "go_stanford_cropped", "go_stanford2"]
MODE = "PER CLASS"
single_class = "Stop"

IMAGE_SIZE = (120, 160)

def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )
def resize_image(image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Resizes an image using Lanczos3 interpolation. Expects & returns uint8."""
    assert image.dtype == tf.uint8
    image = tf.image.resize(image, size, method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image

def read_resize_encode_image(path: str, size: Tuple[int, int]) -> tf.Tensor:
    """Reads, decodes, resizes, and then re-encodes an image."""
    data = tf.io.read_file(path)
    image = tf.image.decode_jpeg(data)
    image = resize_image(image, size)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
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

def process_images(path):
    image_paths = sorted(glob.glob(os.path.join(path, "*.jpg")))
    imgs = [read_resize_encode_image(img_path, IMAGE_SIZE) for img_path in image_paths]
    return imgs

def process_trajs(path):
    # load_data
    fp = os.path.join(path, "traj_data_language.pkl")
    if os.path.exists(fp):

        with open(fp, "rb") as f:
            data = pickle.load(f)
        
        images = process_images(path)
        lang = data["language_instructions"]
        varied_lang = data["varied_language_instructions"]
        CHUNK_SIZE = data["chunk_size"]

        outs = []
        for idx in range(len(lang)):
            out = dict()
            out["lang"] = lang[idx]
            out["varied_lang"] = varied_lang[idx]
            if MODE == "PER CLASS":
                if out["lang"] == single_class:
                    try:
                        out["obs"] = images[idx*CHUNK_SIZE:(idx+1)*CHUNK_SIZE]
                    except: 
                        out["obs"] = images[idx*CHUNK_SIZE:]
                    outs.append(out)
            else:
                try:
                    out["obs"] = images[idx*CHUNK_SIZE:(idx+1)*CHUNK_SIZE]
                except: 
                    out["obs"] = images[idx*CHUNK_SIZE:]
                outs.append(out)
    
        return outs
    else:
        return []

def get_traj_paths(path, train_proportion):
    train_traj = []
    val_traj = []
    dataset_dir = path.split("/")[-1]
    search_path = os.path.join(path, "*")
    all_traj = glob.glob(search_path)
    print("before: ", len(all_traj))
    # if dataset_dir == "sacson":
    #     all_traj = [x for x in all_traj if x.split("/")[-1].split("-")[3] != "cory1"]
    #     print("len after filtering sacson: ")
    #     print(len(all_traj))
    if dataset_dir == "go_stanford_cropped":
        all_traj = [x for x in all_traj if x.split("/")[-1][:3] != "sim"]
        print("len after filtering stanford: ")
        print(len(all_traj))


    if not all_traj: 
        logging.info(f"no trajs found in {search_path}")

    if len(INDOOR_DATASETS) > 1:
        random.shuffle(all_traj)
        if dataset_dir == "sacson" or dataset_dir == "cory_hall" or dataset_dir == "go_stanford2":
            train_traj += all_traj
        else: 
            val_traj += all_traj
            
        np.random.shuffle(train_traj)
        np.random.shuffle(val_traj)
        print("Train: ", len(train_traj))
        print("Val: ", len(val_traj))
    else:
        np.random.shuffle(all_traj)
        train_traj += all_traj[: int(len(all_traj) * train_proportion)]
        val_traj += all_traj[int(len(all_traj) * train_proportion) :]

    return train_traj, val_traj

# create a tfrecord for a group of trajectories
def create_tfrecord(paths, output_path, tqdm_func, global_tqdm):
    writer = tf.io.TFRecordWriter(output_path)
    for path in paths: 
        
        outs = process_trajs(path)

        for out in outs:
            try: 
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            k: tensor_feature(v) for k, v in flatten_dict(out).items()
                        }
                    )
                )
                writer.write(example.SerializeToString())
            except Exception as e:
                import sys
                import traceback

                traceback.print_exc()
                logging.error(f"Error processing {path}")
                sys.exit(1)

        global_tqdm.update(1)
    writer.close()
    global_tqdm.write(f"Finished {output_path}")

def main(_argv):
    assert FLAGS.depth >= 1

    if tf.io.gfile.exists(FLAGS.output_path):
        if FLAGS.overwrite:
            logging.info(f"Deleting {FLAGS.output_path}")
            tf.io.gfile.rmtree(FLAGS.output_path)
        else:
            logging.info(f"{FLAGS.output_path} exists, exiting")
            return
    print(f"USING MODE: {MODE} and CLASS: {single_class}")
    # each path is a directory that contains dated directories
    paths = [os.path.join(FLAGS.input_path, dataset_dir) for dataset_dir in INDOOR_DATASETS]

    # get trajecotry paths in parallel
    with Pool(FLAGS.num_workers) as p:
        train_paths, val_paths = zip(
            *p.map(
                partial(get_traj_paths, train_proportion=FLAGS.train_proportion), paths
            )
        )
    train_paths = [x for y in train_paths for x in y]
    val_paths = [x for y in val_paths for x in y]
    random.shuffle(train_paths)
    random.shuffle(val_paths)

    # shard paths
    train_shards = np.array_split(
        train_paths, np.ceil(len(train_paths) / FLAGS.shard_size)
    )

    # create output paths
    tf.io.gfile.makedirs(os.path.join(FLAGS.output_path, "train"))
    
    train_output_paths = [
        os.path.join(FLAGS.output_path, "train", f"{i}.tfrecord")
        for i in range(len(train_shards))
    ]

    if len(val_paths) != 0:
        val_shards = np.array_split(val_paths, np.ceil(len(val_paths) / FLAGS.shard_size))
        tf.io.gfile.makedirs(os.path.join(FLAGS.output_path, "val"))

        val_output_paths = [
            os.path.join(FLAGS.output_path, "val", f"{i}.tfrecord")
            for i in range(len(val_shards))
            ]
        # create tasks (see tqdm_multiprocess documenation)
        tasks = [
        (create_tfrecord, (train_shards[i], train_output_paths[i]))
        for i in range(len(train_shards))
        ] + [
        (create_tfrecord, (val_shards[i], val_output_paths[i]))
        for i in range(len(val_shards))
        ]
        total_len = len(train_paths) + len(val_paths)

    else:
        tasks = [
            (create_tfrecord, (train_shards[i], train_output_paths[i]))
            for i in range(len(train_shards))
        ]
        total_len = len(train_paths)
    # run tasks
    pool = TqdmMultiProcessPool(FLAGS.num_workers)
    with tqdm.tqdm(
        total=total_len,
        dynamic_ncols=True,
        position=0,
        desc="Total progress",
    ) as pbar:
        pool.map(pbar, tasks, lambda _: None, lambda _: None)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass

