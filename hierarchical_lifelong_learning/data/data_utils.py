import numpy as np
import os
from PIL import Image
from typing import Any, Iterable, Tuple
from functools import partial 
from google.cloud import storage
#import torch
#from torchvision import transforms
#import torchvision.transforms.functional as TF
#import torch.nn.functional as F
import tensorflow as tf
import io
from typing import Union
import dlimp as dl
from dlimp.dataset import DLataset

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training

base_instructions = ["Turn left", "Turn right", "Go forward", "Stop"]

def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:
        print(blob.name)

    return blobs

def get_lifelong_paths(data_path: str, dates: tuple) -> list:

    # Get the paths to the data we want
    paths = []
    prefix = "lifelong_data_"
    folders = list_blobs(data_path)

    for folder in folders:
        date = ("_").join(folder.name.split("_")[2:])
        if date > dates[0] and date < dates[1]:
            paths.append(folder.path)
    return paths

def make_dataset(
    name: str,
    data_path: str,
    dates: tuple,
    image_size: int,
) -> dl.DLataset:

    paths = get_lifelong_paths(data_path, dates)
    dataset = dl.Dataset.from_tfrecords(paths).iterator()
    breakpoint()

    for sample in dataset: 
        print(sample)
    
    # dataset = (
    #     dl.DLataset.from_tfrecords(paths)
    #     .map(dl.transforms.unflatten_dict)
    #     .map(getattr(Transforms, name))
    #     .filter(lambda x: tf.math.reduce_all(x["lang"] != ""))
    #     .apply(
    #         partial(
    #             getattr(goal_relabeling, goal_relabeling_fn), **goal_relabeling_kwargs
    #         ),
    #     )
    #     .unbatch()
    #     .shuffle(shuffle_buffer_size)
    # )

    # dataset = dataset.map(
    #     partial(dl.transforms.decode_images, match=["curr", "goals", "subgoals"])
    # ).map(
    #     partial(
    #         dl.transforms.resize_images,
    #         match=["curr", "goals", "subgoals"],
    #         size=(image_size, image_size),
    #     )
    # )

    # if train:
    #     dataset = dataset.map(
    #         partial(
    #             dl.transforms.augment,
    #             traj_identical=False,
    #             keys_identical=True,
    #             match=["curr", "goals", "subgoals"],
    #             augment_kwargs=augment_kwargs,
    #         )
    #     )

    # # normalize images to [-1, 1]
    # dataset = dataset.map(
    #     partial(
    #         dl.transforms.selective_tree_map,
    #         match=["curr", "goals", "subgoals"],
    #         map_fn=lambda v: v / 127.5 - 1.0,
    #     )
    # )

    return dataset.repeat()

def relabel_primitives(traj, chunk_size, yaw_threshold, pos_threshold):
    traj_obs = traj["observation"]
    yaw = traj_obs["yaw"]
    pos = traj_obs["position"]
    gt_lang = traj_obs["gt_lang"]
    print(traj["_len"]) 
    if yaw.shape[0] == None: 
        print("No yaw values")
        return traj
    else:
        print("Len of traj is: ", yaw.shape)
    num_chunks = len(yaw)//chunk_size
    yaw_chunks = np.split(yaw, num_chunks)
    pos_chunks = np.split(pos, num_chunks)
    image_chunks = np.split(traj_obs["obs"], num_chunks)
    gt_lang = traj[gt_lang][-1]
    goal = traj_obs["goal"][-1]
    print(gt_lang)
    samples_out = []

    for yaw_chunk, pos_chunk, image_chunk in zip(yaw_chunks, pos_chunks, image_chunks):

        yaw_delta = float(get_yaw_delta(yaw_chunk).squeeze())
        pos_delta = np.sqrt(np.sum(np.square(pos_chunk[-1,:] - pos_chunk[0,:]), axis=-1))

        if yaw_delta > yaw_threshold:
            lang = base_instructions[0]
            varied_lang = random.choice(varied_left)
        elif yaw_delta < -yaw_threshold:
            lang = base_instructions[1]
            varied_lang = random.choice(varied_right)
        else:
            if pos_delta > pos_threshold:
                lang = base_instructions[2]
                varied_lang = random.choice(varied_forward)
            else:
                lang = base_instructions[3]
                varied_lang = random.choice(varied_stop)
        sample = {}
        sample["obs"] = image_chunk
        sample["lang"] = lang
        sample["varied_lang"] = varied_lang
        sample["gt_lang"] = gt_lang 
        sample["goal"] = goal

        samples_out.append(sample)

    data = tf.concat(samples_out, axis=0)
    return data

def relabel_vlm(dataset):
    ''' Relabel the dataset with the VLM instructions'''
    # Should have available list of skills ? 
    # Otherwise just use VLM to get an idea of what happened
    pass

def get_yaw_delta(yaw_reshape):
    yaw_delta = yaw_reshape[:,-1] - yaw_reshape[:,0]
    yaw_delta_sign = np.where(yaw_delta >= np.pi, -1, 0)
    yaw_delta_sign = np.where(yaw_delta < -np.pi, 1, yaw_delta_sign)
    yaw_delta = yaw_delta + yaw_delta_sign*2*np.pi
    return yaw_delta

def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    data_ext = {
        "image": ".jpg",
        # add more data types here
    }
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)


#def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
#    """
#    Calculate deltas between waypoints
#
#    Args:
#        waypoints (torch.Tensor): waypoints
#    Returns:
#        torch.Tensor: deltas
#    """
#    num_params = waypoints.shape[1]
#    origin = torch.zeros(1, num_params)
#    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
#    deltas = waypoints - prev_waypoints
#    if num_params > 2:
#        return calculate_sin_cos(deltas)
#    return deltas


#def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
#    """
#    Calculate sin and cos of the angle
#
#    Args:
#        waypoints (torch.Tensor): waypoints
#    Returns:
#        torch.Tensor: waypoints with sin and cos of the angle
#    """
#    assert waypoints.shape[1] == 3
#    angle_repr = torch.zeros_like(waypoints[:, :2])
#    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
#    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
#    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


#def transform_images(
#    img: Image.Image, transform: transforms, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
#):
#    w, h = img.size
#    if w > h:
#        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
#    else:
#        img = TF.center_crop(img, (int(w / aspect_ratio), w))
#    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)
#    viz_img = TF.to_tensor(viz_img)
#    img = img.resize(image_resize_size)
#    transf_img = transform(img)
#    return viz_img, transf_img


#def resize_and_aspect_crop(
#    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
#):
#    w, h = img.size
#    if w > h:
#        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
#    else:
#        img = TF.center_crop(img, (int(w / aspect_ratio), w))
#    img = img.resize(image_resize_size)
#    resize_img = TF.to_tensor(img)
#    return resize_img


#def img_path_to_data(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
#    """
#    Load an image from a path and transform it
#    Args:
#        path (str): path to the image
#        image_resize_size (Tuple[int, int]): size to resize the image to
#    Returns:
#        torch.Tensor: resized image as tensor
#    """
#    # return transform_images(Image.open(path), transform, image_resize_size, aspect_ratio)
#    return resize_and_aspect_crop(Image.open(path), image_resize_size)  

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
