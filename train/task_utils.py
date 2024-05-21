from agentlace.trainer import TrainerConfig
import tensorflow as tf
import numpy as np 


def make_trainer_config():
    return TrainerConfig(
        port_number=5488,
        broadcast_port=5489,
        request_types=["send-stats", "get-model-config"],
    )

def observation_format():
    return {
        "obs": tf.TensorSpec((), tf.string, name="image"),
        "position": tf.TensorSpec((2,), tf.float64, name="position"),
        "yaw": tf.TensorSpec((), tf.float64, name="orientation"),
    }

def robot_data_format():
    return {
        "observation": observation_format(),
        "action": tf.TensorSpec((2,), tf.float64, name="action"),
    }

def rlds_data_format():
    return {
        "observation": observation_format(),
        "goal": tf.TensorSpec((), tf.string, name="image"),
        "action": tf.TensorSpec((2,), tf.float64, name="action"),
        "is_first": tf.TensorSpec((), tf.bool, name="is_first"),
        "is_last": tf.TensorSpec((), tf.bool, name="is_last"),
        "is_terminal": tf.TensorSpec((), tf.bool, name="is_terminal"),
        "status": tf.TensorSpec((), tf.string, name="status"),
}

def task_data_format():
    return {
        # **robot_data_format(),
        "observation": {
            **observation_format(),
            "goal": {
                "image": tf.TensorSpec((), tf.string, name="image"),
                "position": tf.TensorSpec((3,), tf.float64, name="position"),
                "orientation": tf.TensorSpec((4,), tf.float64, name="orientation"),
                "reached": tf.TensorSpec((), tf.bool, name="reached"),
            },
        },
        "action": tf.TensorSpec((2,), tf.float64, name="action"),
        "is_first": tf.TensorSpec((), tf.bool, name="is_first"),
        "is_last": tf.TensorSpec((), tf.bool, name="is_last"),
        "is_terminal": tf.TensorSpec((), tf.bool, name="is_terminal"),
    }