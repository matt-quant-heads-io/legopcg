"""Data Loader"""

import os

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

import random
import numpy as np
import pandas as pd

from utils.utils import get_curr_obs_coords_lst_from_model_map_lst, get_template_from_model_map_lst, write_obs_to_mpd_file


class DataLoader:
    """Data Loader class"""

    def generate_data(self, data_config):
        move_inc = data_config['move_inc']
        path = data_config['goals_path']
        curr_obs_coords_lst = get_curr_obs_coords_lst_from_model_map_lst(path)
        template_model_map_lst = get_template_from_model_map_lst(path)
        MODEL_CHG_TO_DISCRETE_ACTION_MAP = {
            (0, move_inc): 0,
            (1, move_inc): 1,
            (2, move_inc): 2,
            (0, -move_inc): 3,
            (1, -move_inc): 4,
            (2, -move_inc): 5,
        }
        obs_map = {f"col_{i}":[] for i in range(len(curr_obs_coords_lst)*2*3)}
        obs_map["target"] = []

        num_episodes = data_config['num_gen_episodes']
        for i in range(num_episodes):
            curr_idx = 0
            curr_map = [[0,0,0] for _ in range(len(curr_obs_coords_lst)*2)]
            direction = None
            change = None
            copy_curr_obs_coords_lst = curr_obs_coords_lst

            for idx, coords in enumerate(copy_curr_obs_coords_lst):
                curr_map = [[0,0,0] for _ in range(len(copy_curr_obs_coords_lst)*2)]
                curr_obs_map = copy_curr_obs_coords_lst

                direction = random.choice([0, 1, 2])
                change = random.choice([move_inc, -move_inc])
                coords[direction] += change
                curr_map[idx: len(copy_curr_obs_coords_lst)+idx] = copy_curr_obs_coords_lst

                for i, val in enumerate(np.array(curr_map).flatten()):
                    obs_map[f"col_{i}"].append(val)

                obs_map["target"].append(MODEL_CHG_TO_DISCRETE_ACTION_MAP[(direction, change*(-1))])


            df = pd.DataFrame(obs_map)
            df.to_csv(f"{data_config['output_path']}/traj{i}.csv", index=False)

            obs_map = {f"col_{i}":[] for i in range(len(curr_obs_coords_lst)*2*3)}
            obs_map["target"] = []

        write_path = f"{data_config['output_path']}/destroyed_model_1.mpd"
        write_obs_to_mpd_file(write_path, curr_obs_coords_lst, template_model_map_lst)


    @staticmethod
    def load_data(data_config):
        """Loads dataset from path"""
        pod_root_path = data_config['path']

        dfs = []
        X = []

        for file in os.listdir(pod_root_path):
            if not file.endswith(".csv"):
                continue
            df = pd.read_csv(f"{pod_root_path}/{file}")
            dfs.append(df)

        df = pd.concat(dfs)

        df = df.sample(frac=1).reset_index(drop=True)
        y_true = df[["target"]]
        y = to_categorical(y_true)
        df.drop("target", axis=1, inplace=True)
        y = y.astype("int32")


        for idx in range(len(df)):
            x = df.iloc[idx, :].values.astype("float32")#.reshape((1, 234))
            X.append(x)

        X = np.array(X)

        return X, y

    @staticmethod
    def preprocess_data(dataset, batch_size, buffer_size, image_size):
        """ Preprocess and splits into training and test"""

        train = dataset['train'].map(lambda image: DataLoader._preprocess_train(image, image_size),
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = dataset['test'].map(lambda image: DataLoader._preprocess_test(image, image_size))

        train_dataset = train.shuffle(buffer_size).batch(batch_size).cache().repeat()
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_dataset = test.batch(batch_size)

        return train_dataset, test_dataset

    @staticmethod
    def _preprocess_train(datapoint, image_size):
        """ Loads and preprocess  a single training image """
        input_image = tf.image.resize(datapoint['image'], (image_size, image_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (image_size, image_size))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = DataLoader._normalize(input_image, input_mask)

        return input_image, input_mask

    @staticmethod
    def _preprocess_test(datapoint, image_size):
        """ Loads and preprocess a single test images """

        input_image = tf.image.resize(datapoint['image'], (image_size, image_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (image_size, image_size))

        input_image, input_mask = DataLoader._normalize(input_image, input_mask)

        return input_image, input_mask

    @staticmethod
    def _normalize(input_image, input_mask):
        """ Normalise input image
        Args:
            input_image (tf.image): The input image
            input_mask (int): The image mask
        Returns:
            input_image (tf.image): The normalized input image
            input_mask (int): The new image mask
        """
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask
