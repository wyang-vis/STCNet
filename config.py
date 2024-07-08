#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:35:48 2019

@author: aditya
"""

r"""This module provides package-wide configuration management."""
from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        ALPHA: 1000.0
        BETA: 0.5

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048, "BETA", 0.7])
    >>> _C.ALPHA  # default: 100.0
    1000.0
    >>> _C.BATCH_SIZE  # default: 256
    2048
    >>> _C.BETA  # default: 0.1
    0.7

    Attributes
    ----------
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.MODE = 'global'
        self._C.MODEL.SESSION = 'ps128_bs1'

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 1
        self._C.OPTIM.NUM_EPOCHS = 100
        self._C.OPTIM.NEPOCH_DECAY = [100]
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 0.0002
        self._C.OPTIM.BETA1 = 0.5

        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.RESUME = True
        self._C.TRAINING.SAVE_IMAGES = False
        self._C.TRAINING.SAVE_DIR = 'checkpoints'
        self._C.TRAINING.TRAIN_PS = 64
        self._C.TRAINING.VAL_PS = 64

        self._C.TESTING = CN()
        self._C.TESTING.RESUME = True

        self.result_dir='./output'


        self.father_train_path_npz='./Datasets/GROPO/train'
        self.father_val_path_npz='./Datasets/GROPO/test'
        self.father_test_path_npz='./Datasets/GROPO/test'



        self.train_iters=1000
        self.unrolling_len=3
        self.pair=True
        self.geometry_aug=False
        self.num_bins=6
        self.future_frames = 1
        self.past_frames = 1


        self.skip_type='sum'  # sum or concat=skip_concat
        self.activation='relu' ##sigmoid  or relu
        self.recurrent_block_type='convlstm'  # 可以调整
        self.num_encoders=4
        self.base_num_channels=16
        self.use_upsample_conv=False
        self.norm='BN'
        self.rec_channel=1
        self.num_residual_blocks=2
        self.num_output_channels=3
        self.hot_pixels_file=None
        self.norm_method='normal'
        self.no_normalize=False
        self.flip=False
        self.VGGLayers=[1,2,3,4]
        self.w_VGG=0.2
        ##__________________________________________________
        self.rgb_range=255
        self.n_resblocks =19
        self.n_feats =64
        self.kernel_size =5
        self.n_scales =1

        # self.specify_DVS_DV=
        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
