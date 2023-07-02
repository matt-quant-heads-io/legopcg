import pandas as pd

from .base_dataloader import BaseDataLoader

import os
import random

from keras.layers import Input, Dense, Conv2D, Concatenate, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.utils import np_utils

import tensorflow as tf

from tensorflow.keras.layers import Dense
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, Flatten, Concatenate
from keras.optimizers import SGD
from keras.models import Model

import os
import re
import glob
import numpy as np
import tensorflow.keras.backend as K


class PoDCNNDataLoader(BaseDataLoader):
    def __init__(self):
        pass

    @staticmethod
    def get_trainer_id():
        return "podcnn_trainer"

    def load_data(self, config):
        train_data_path = config.data["train_data_path"]
        obs_size = config.data["obs_size"]
        use_signed_inputs = config.data["use_signed_inputs"]

        dfs = []
        X = []

        for file in os.listdir(train_data_path):
            print(f"compiling df {file}")
            if file.endswith(".ipynb_checkpoints"):
                continue
            df = pd.read_csv(f"{train_data_path}/{file}")
            dfs.append(df)

        df = pd.concat(dfs)
        df = df[:300000]

        df = df.sample(frac=1).reset_index(drop=True)
        y_true = df[["target"]]
        y = np_utils.to_categorical(y_true)
        df.drop("target", axis=1, inplace=True)
        y = y.astype("int32")
        df["num_lego_pieces_input_target"] -= 1

        num_lego_pieces_input_target = round(
            df[["num_lego_pieces_input_target"]] / 27.0,
            2,
        )
        df.drop("num_lego_pieces_input_target", axis=1, inplace=True)

        if use_signed_inputs:
            num_lego_pieces_signed = np_utils.to_categorical(
                df[["num_lego_pieces_signed"]] + 1
            )
        else:
            num_lego_pieces_signed = (df[["num_lego_pieces_signed"]] + 1) / 3.0

        df.drop("num_lego_pieces_signed", axis=1, inplace=True)

        cond_input_target = np.column_stack((num_lego_pieces_input_target,))
        print(f"cond_input_target: {cond_input_target.shape}")
        print(f"cond_input_target: {cond_input_target[0]}")
        signed_output = np.column_stack((num_lego_pieces_signed,))
        print(f"signed_output: {signed_output.shape}")
        print(f"signed_output: {signed_output[0]}")

        action_dim = 37
        obs_size = 6

        for idx in range(len(df)):
            x = df.iloc[idx, :].values.astype("int32")
            row = []
            for val in x:
                # print(f"val: {val}")
                oh = [0] * 37
                oh[val] = 1
                row.append(oh)
            X.append(np.array(row).reshape((obs_size, obs_size, obs_size, action_dim)))

        X = np.array(X)

        print(f"X: {X.shape}")
        print(f"cond_input_target: {cond_input_target.shape}")
        print(f"y: {y.shape}")

        # TODO: switch to use this (with signed inputs after confirm MLP Block is trained per signed output)
        # return [K.constant(X), K.constant(np.array(cond_input_target))], [
        #     np.array(signed_output),
        #     y,
        # ]
        return [
            K.constant(X),
            K.constant(np.array(cond_input_target)),
            K.constant(np.array(signed_output)),
        ], y
