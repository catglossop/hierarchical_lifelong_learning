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
import cv2
import glob
import logging
import os
import pickle
import random
from datetime import datetime
from functools import partial
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from tqdm_multiprocess import TqdmMultiProcessPool
import pandas as pd
import h5py
import dlimp as dl
from dlimp.utils import resize_image, tensor_feature
FLAGS = flags.FLAGS
flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.95, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_integer("shard_size", 200, "Maximum number of trajectories per shard")
IMAGE_SIZE = (256, 256)
CAMERA_VIEWS = ["cam_high", "cam_low"]
def resize_encode_image(image, size) -> tf.Tensor:
    """Reads, decodes, resizes, and then re-encodes an image."""
    image = resize_image(image, size)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return tf.io.encode_jpeg(image, quality=95)
def process_images(hf_images):  # processes images at a trajectory level
    d = dict()
    for i, k in enumerate(CAMERA_VIEWS):
        images = []
        for img in hf_images[k]:
            img = cv2.imdecode(img, 1)
            img = tf.convert_to_tensor(img[:,:,::-1], dtype=tf.uint8)
            images.append(resize_encode_image(img, IMAGE_SIZE)[None])
        d[f'images{i}'] = tf.concat(images, axis=0)
    return d
def process_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"]
def process_actions(path):
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list
def process_lang(path):
    fp = os.path.join(path, "lang.txt")
    text = ""  # empty string is a placeholder for missing text
    if os.path.exists(fp):
        with open(fp, "r") as f:
            text = f.readline().strip()
    return text
# create a tfrecord for a group of trajectories
def create_tfrecord(paths, output_path, lang, tqdm_func, global_tqdm):
    writer = tf.io.TFRecordWriter(output_path)
    for path in paths:
        try:
            out = dict()
            with h5py.File(path, 'r') as hf:
                out["obs"] = process_images(hf['observations']['images'])
                out["obs"]["state"] = hf['observations']['qpos'][:].astype(np.float32).copy()
                out["actions"] = hf['action'][:].astype(np.float32).copy()
                out["lang"] = lang
            assert (
                len(out["actions"])
                == len(out["obs"]["state"])
                == len(out["obs"]["images0"])
            )
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        k: tensor_feature(v)
                        for k, v in dl.transforms.flatten_dict(out).items()
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
def main(_):
    if tf.io.gfile.exists(FLAGS.output_path):
        if FLAGS.overwrite:
            logging.info(f"Deleting {FLAGS.output_path}")
            tf.io.gfile.rmtree(FLAGS.output_path)
        else:
            logging.info(f"{FLAGS.output_path} exists, exiting")
            return
    # each path is a directory that contains dated directories
    paths = glob.glob(os.path.join(FLAGS.input_path, '*.hdf5'))
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
    lang = "cut the dough on the white board."  # "cut the sushi and place it on the pink plate."
    tasks = [
        (create_tfrecord, (train_shards[i], train_output_paths[i], lang))
        for i in range(len(train_shards))
    ] + [
        (create_tfrecord, (val_shards[i], val_output_paths[i], lang))
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
if _name_ == "_main_":
    app.run(main)
