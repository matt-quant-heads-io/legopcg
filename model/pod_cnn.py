# -*- coding: utf-8 -*-
"""PoDCNN model"""

# standard library

# external
import tensorflow as tf

from tensorflow.keras.layers import Dense
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.models import Model

# internal
from .base_model import BaseModel
from dataloader.dataloader import DataLoader
# from utils.logger import get_logger
from executor.podcnn_trainer import PoDCNNTrainer

# TODO: implement this
LOG = None #get_logger('PoDCNN')


class PoDCNN(BaseModel):
    """PoDCNN Model Class"""

    def __init__(self, config):
        super().__init__(config)
        self.output_channels = self.config['model']['output']

        self.dataset = None
        self.info = None
        self.train_config = self.config['train']
        self.data_config = self.config['data']
        self.model_config = self.config['model']
        self.batch_size = self.train_config['batch_size']
        self.buffer_size = self.train_config['buffer_size']
        self.epochs = self.train_config['epochs']


        self.num_classes = self.data_config['num_classes']
        self.val_subsplits = self.train_config['val_subsplits']
        self.validation_steps = 0
        self.train_length = 0
        self.steps_per_epoch = self.train_config['steps_per_epoch']
        self.saved_model_path = f"{self.model_config['saved_model_path']}/{self.model_config['saved_model_name']}"

        self.dims = self.data_config['dims']
        self.obs_size = self.data_config['obs_size']
        self.action_dim = self.data_config['action_dim']

        self.train_dataset = []
        self.target_dataset = []
        self.test_dataset = []

        self.model = self.build()

    def generate_data(self):
        """Generates and persists data and Preprocess data """
        # LOG.info(f'Generating data for PoDCNN model...')

        DataLoader().generate_data(self.data_config)


    def load_data(self):
        """Loads and Preprocess data """
        # LOG.info(f'Loading {self.config.data.path} dataset...')
        self.train_dataset, self.target_dataset = DataLoader().load_data(self.data_config)
        # self.train_dataset, self.test_dataset = DataLoader.preprocess_data(self.dataset, self.batch_size,
        #                                                                    self.buffer_size, self.image_size)
        self._set_training_parameters()

    def _set_training_parameters(self):
        """Sets training parameters"""
        # self.train_length = self.info.splits['train'].num_examples
        # self.steps_per_epoch = self.train_length // self.batch_size
        # self.validation_steps = self.info.splits['test'].num_examples // self.batch_size // self.val_subsplits
        pass

    def build(self):
        """ Builds the Keras model based """
        inputs = [
            Input(shape=(self.obs_size, self.obs_size, self.action_dim))
        ]

        x = Conv2D(
            128,
            (3, 3),
            activation="relu",
            input_shape=(self.obs_size, self.obs_size, self.action_dim),
            padding="SAME",
        )(inputs[0])

        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(128, (3, 3), activation="relu", padding="SAME")(x)
        x = Conv2D(256, (3, 3), activation="relu", padding="SAME")(x)
        x = Flatten()(x)

        output = [
            Dense(self.action_dim, activation="softmax")(x),    
        ]

        conditional_cnn_model = Model(inputs, output)

        # LOG.info('Model was built successfully')

        return conditional_cnn_model 

    def train(self):
        """Compiles and trains the model"""
        # LOG.info('Training started')

        optimizer = SGD()
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.CategoricalAccuracy()]

        trainer = PoDCNNTrainer(self.model, self.train_dataset, self.target_dataset, loss, optimizer, metrics, self.epochs, self.steps_per_epoch, self.saved_model_path)
        trainer.train()

    def _run_inference(self):
        pass

    def evaluate(self):
        """Predicts resuts for the test dataset"""
        self._run_inference()


        