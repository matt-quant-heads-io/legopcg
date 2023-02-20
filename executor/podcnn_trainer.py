import os

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# from utils.logger import get_logger

# LOG = get_logger('trainer')


class PoDCNNTrainer:

    def __init__(self, model, train_data, train_targets, loss_fn, optimizer, metrics, epochs, steps_per_epoch, saved_model_path):
        self.model = model
        self.input = input
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_data = train_data
        self.train_targets = train_targets
        self.saved_model_path = saved_model_path

        # self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        # self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, './tf_ckpts', max_to_keep=3)

        # self.train_log_dir = 'logs/gradient_tape/'
        # self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)

        self.model_save_path = saved_model_path

    def train(self):
        """Compiles and trains the model"""
        # LOG.info('Training started')

        mcp_save = ModelCheckpoint(
            self.saved_model_path,
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

        self.model.fit(
            self.train_data,
            self.train_targets,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            verbose=2,
            callbacks=[mcp_save],
        )