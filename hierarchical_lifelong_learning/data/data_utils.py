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
import tensorflow_datasets as tfds
import io
from typing import Union
import dlimp as dl
from dlimp.dataset import DLataset
from openai import OpenAI

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training

client = OpenAI()

base_instructions = ["Turn left", "Turn right", "Go forward", "Stop"]

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
            if folder.name.endswith(".tfrecord-00000"):
                new_path = os.path.join("gs://", data_path, folder.name)
                print(new_path)
                paths.append(new_path)
            
    return paths

def make_dataset(
    name: str,
    data_path: str,
    dates: tuple,
    image_size: int,
) -> dl.DLataset:

    paths = get_lifelong_paths(data_path, dates)
    breakpoint()
    dataset = dl.DLataset.from_tfrecords(paths)
    #builder = tfds.builder("lifelong_data", try_gcs=True, data_dir=path)
    #print(path)
    #subdataset = dl.DLataset.from_rlds(builder)
    #if dataset is None: 
    #    dataset = subdataset
    #else:
    #    dataset = dataset.concatenate(subdataset)
    #breakpoint()

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

def relabel_primitives(traj, primitive, chunk_size, yaw_threshold, pos_threshold):
    yaw = traj["yaw"]
    pos = traj["position"]
    if yaw.shape[0] < chunk_size: 
        yaw_chunks = [yaw]
        pos_chunks = [pos]
        image_chunks = [traj["obs"]]
    else:
        num_chunks = len(yaw)//chunk_size
        yaw_chunks = np.array_split(yaw, num_chunks)
        pos_chunks = np.array_split(pos, num_chunks)
        image_chunks = [[traj["obs"][i*chunk_size:(i+1)*chunk_size]] for i in range(0, len(traj["obs"]), chunk_size)]
    gt_lang = traj["gt_lang"]
    goal = traj["goal"]
    samples_out = []

    for yaw_chunk, pos_chunk, image_chunk in zip(yaw_chunks, pos_chunks, image_chunks):
        yaw_delta = float(get_yaw_delta(yaw_chunk).squeeze())
        pos_delta = np.sqrt(np.sum(np.square(pos_chunk[-1,:] - pos_chunk[0,:]), axis=-1))
        # print("Yaw delta: ", yaw_delta)
        # print("Pos delta: ", pos_delta)
        if yaw_delta > yaw_threshold:
            lang = base_instructions[0]
            #varied_lang = random.choice(varied_left)
        elif yaw_delta < -yaw_threshold:
            lang = base_instructions[1]
            #varied_lang = random.choice(varied_right)
        else:
            if pos_delta > pos_threshold:
                lang = base_instructions[2]
                #varied_lang = random.choice(varied_forward)
            else:
                lang = base_instructions[3]
                #varied_lang = random.choice(varied_stop)
        sample = {}
        sample["obs"] = image_chunk
        sample["lang"] = lang
        #sample["varied_lang"] = varied_lang
        sample["gt_lang"] = gt_lang 
        sample["goal"] = goal
        # print("Relabelled lang: ", lang)
        if primitive == "all":
            samples_out.append(sample)
        elif primitive == base_instructions[0] and lang == base_instructions[0]:
            samples_out.append(sample)
        elif primitive == base_instructions[1] and lang == base_instructions[1]:
            samples_out.append(sample)
        elif primitive == base_instructions[2] and lang == base_instructions[2]:
            samples_out.append(sample)
        elif primitive == base_instructions[2] and lang == base_instructions[2]:
            samples_out.append(sample)

    return samples_out

def tensor_to_base64(tensor):
    image = tensor.permute(1, 2, 0).numpy()
    pil_image = Image.fromarray((image * 255).astype('uint8'))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def relabel_vlm(dataset, chunk_size = 5):
    def analyze_images_with_vlm(images):
        prompt = [{"type": "text", "text": "These images represent a trajectory of robot visual observations:"}]
        
        for i, image in enumerate(images):
            image_base64 = tensor_to_base64(image)
            prompt.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                },
            })
        
        question = """ Given these series of frames, construct a descriptive label which describes the trajecotry the robot has taken and where it ends
                        Return the label which is in the form 'go to the x' where x is a descriptive landmark in the last frame.
                        Only return the final label."""
        prompt.append({"type": "text", "text": question})

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        label = completion.choices[0].message.content
        return label

    labels = []
    for start_idx in range(0, len(dataset)):
        end_idx = min(start_idx + chunk_size, len(dataset))
        obs_images_list = []
        
        for i in range(start_idx, end_idx):
            datapoint = dataset[i]
            obs_images, goal_image, *rest = data_point
            obs_images_list.append(obs_images)
        
        images = obs_images_list[i]
        new_label = analyze_images_with_vlm(obs_images_list)
        labels.append(new_label)    
        
    return relabeled_dataset

def get_yaw_delta(yaw_reshape):
    yaw_delta = yaw_reshape[-1] - yaw_reshape[0]
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
