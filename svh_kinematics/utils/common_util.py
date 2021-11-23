import numpy as np
import random
import os


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def deg2rad(angel_list):
    joint_angel_deg = np.asarray(angel_list)
    joint_angel_rad = np.deg2rad(joint_angel_deg)
    return joint_angel_rad.tolist()
