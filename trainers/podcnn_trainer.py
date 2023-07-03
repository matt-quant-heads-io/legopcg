import os
import json

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.models import Model

from .base_trainer import BaseTrainer
from models import MODELS_MAP
from models.podcnn_model import PoDCNNModel

from dataloaders import DATALOADERS_MAP
from dataloaders.podcnn_dataloader import PoDCNNDataLoader

# TODO: implement logging below
# from utils.logger import get_logger
# LOG = get_logger('trainer')


class PoDCNNTrainer(BaseTrainer):
    def __init__(self, config_id):
        super().__init__(config_id)
        self.model = MODELS_MAP[PoDCNNModel.get_trainer_id()]().build(
            self.config
        )  # TODO: change self.config to pass in only model part of self.config
        self.dataloader = DATALOADERS_MAP[PoDCNNDataLoader.get_trainer_id()]()

        print(f"self.config.train: {self.config.train}")
        self.epochs = self.config.train["epochs"]
        self.steps_per_epoch = self.config.train["steps_per_epoch"]
        self.model_save_path = self.config.model["model_save_path"]
        self.model_name = self.config.model["model_name"]

        self.optimizer = SGD()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.metrics = [tf.keras.metrics.CategoricalAccuracy()]

    @staticmethod
    def get_id():
        return "podcnn_trainer"

    def train(self):
        """Compiles and trains the model"""
        # LOG.info('Training started')

        mcp_save = ModelCheckpoint(
            f"{self.model_save_path}/{self.model_name}",
            save_best_only=True,
            monitor="categorical_accuracy",
            mode="max",
        )

        # Configure the model and start training
        self.model.compile(
            loss=self.loss_fn,
            optimizer=self.optimizer,
            metrics=[m for m in self.metrics],
        )

        train_data, train_targets = self.dataloader.load_data(self.config)

        self.model.fit(
            train_data,
            train_targets,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            verbose=2,
            callbacks=[mcp_save],
        )
