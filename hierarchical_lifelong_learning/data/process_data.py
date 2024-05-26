import numpy as np
import os
from PIL import Image
from typing import Any, Iterable, Tuple
from functools import partial 
from google.cloud import storage

import tensorflow as tf
import io
from typing import Union
import dlimp as dl 
from dlimp.dataset import DLataset

from hierarchical_lifelong_learning.data.data_utils import (
    make_dataset,
)


def process_data():
    name = "lifelong_data"
    dates = ('05-24-2024_00-00-00', '05-26-2024_00-00-00')
    data_path = "gs://catg_central2/lifelong"
    image_size = 256

    dataset = make_dataset(name, data_path, dates, image_size)

    return dataset



