import tensorflow as tf

from tensorflow.keras.layers import Dense
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.models import Model

import pandas as pd

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
from gym_pcgrl import wrappers

import tensorflow.keras.backend as K


from .base_model import BaseModel

# TODO: implement this
LOG = None  # get_logger('PoDCNN')


class Linear(tf.keras.layers.Layer):
    def __init__(self, units, name, **kwargs):
        print(**kwargs)
        super().__init__(**kwargs)
        self.units = units
        # self.name = name
        self.i = name

    def build(self, input_shape):
        self.i += 1
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name=f"weight_{self.i}",
        )
        self.i += 1
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
            name=f"b_{self.i}",
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config().copy()
        config.update({"units": self.units, "name": self.name})
        return config


# FOR SIGNED
class MLPCountingBlockSigned(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.linear_1 = Linear(32, 5)
        self.linear_2 = Linear(32, 5)
        self.dense_1 = Dense(3, activation="softmax")
        self.loss = tf.keras.losses.CategoricalCrossentropy(name="counting_head_loss")

    def call(self, inputs):
        x = Concatenate()(inputs[:2])
        x = self.linear_1(x)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        output = self.dense_1(x)

        # cat_cross_entry = self.loss.update_state(inputs[2], output)
        # self.add_loss(cat_cross_entry)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({"name": self.name})
        return config


class PoDCNNModel(BaseModel):
    def __init__(
        self,
    ):
        super().__init__()

    @staticmethod
    def get_trainer_id():
        return "podcnn_trainer"

    def build(self, config):
        """Builds the Keras model based"""
        obs_size = config.model["obs_size"]
        action_dim = config.model["action_dim"]
        inputs = [
            Input(shape=(obs_size, obs_size, obs_size, action_dim)),
            Input(shape=(1,)),
            Input(shape=(3,)),
        ]

        x = Conv3D(
            128,
            (3, 3, 3),
            activation="relu",
            input_shape=(obs_size, obs_size, obs_size, action_dim),
            padding="SAME",
        )(inputs[0])

        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = Conv3D(128, (3, 3, 3), activation="relu", padding="SAME")(x)
        x = Conv3D(256, (3, 3, 3), activation="relu", padding="SAME")(x)
        convolved_features = Flatten()(x)

        x_lego_blocks = MLPCountingBlockSigned(name="lego_pieces_counting_block")(
            [convolved_features, inputs[1], inputs[2]]
        )

        # counting_head = Model(
        #     inputs=inputs, outputs=x_lego_blocks, name="counting_head"
        # )

        x = Concatenate()([convolved_features, x_lego_blocks])
        x = Dense(128)(x)

        output = [
            Dense(action_dim, activation="softmax")(x),
        ]

        conditional_cnn_model = Model(inputs, output)

        # LOG.info('Model was built successfully')

        return conditional_cnn_model
