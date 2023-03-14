# legopcg

This project is a lego building environment for analyzing and experimenting with various generative learning algorithms.


<img src="media/PoD_generates_vintage_car.gif" width="450"/> <img src="media/Controllable_racer.gif" width="450"/>


# Using legopcg

## Step 1: Setup

1. Create a conda environemnt

```
conda create -n legopcg python=3.7 
conda activate legopcg
``` 
2. Install dependencies

```python -m pip install -r requirements.txt```

## Step 2: Defining a config

In the **configs** directory, define a new config file. For example, create an example_config.py and paste in the following code,

```
PODCNN_CONFIG = {
    "data": {
        # ... parameters related to your data go here ...
        "path": f"{pathlib.Path(os.getcwd()).parents[0]}/legopcg/data/input/podcnn/vintage_car",
        "image_size": 128,
        "load_with_info": True,
        "num_classes": 4,
        "num_gen_episodes": 5000,
        "output_path": f"{pathlib.Path(os.getcwd()).parents[0]}/legopcg/data/output",
        "action_dim": 4
    },
    "train": {
        # ... parameters related to your training code go here ...
        "batch_size": 64,
        "buffer_size": 1000,
        "epochs": 500,
        "steps_per_epoch": 128,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
      # ... parameters related to your model code go here ...
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3,
        "saved_model_path": f"{pathlib.Path(os.getcwd()).parents[0]}/legopcg/saved_models",
        "saved_model_name": f"pod_cnn_{datetime.datetime.now()}.h5"

    }
}
```

Next, register the config by accessing configs/__init__.py and adding the following code,

```
import sys,os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# add your config import here after the other config imports
from .example_config import EXAMPLE_CONFIG

CONFIGS_MAP = {
    
    # add your key, value pair at the end of the map
    "example_config": EXAMPLE_CONFIG
}
```


## Step 3: Defining a model
All models inherit from the BaseModel class in model/base_model.py. Therefore, your model must implement the methods

```
class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
```


Here is an example convolutional neural network path-of-destruction model,

```
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
```


## Step 4: Defining a dataloader
For non-RL settings, a dataloader should be used for loading data. For supervised learning problems, a dataloader object can also be defined to generate data.

## Step 5: Define a Trainer
Define a trainer that handles training the self.model member variable in the Model that you defined. Below is the example used for the PoDCNN model,

```
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
```

## Step 6: Run training
Now you can train a model. navigate to the project root (in a terminal) and run main.py via the following command,

```
python main.py --mode train --config_id podcnn_config --gen_train_data
```

Note that if you don't need to generate training data you can also run like so

```
python main.py --mode train --config_id podcnn_config
```

## Step 7: Run inference
That's it! Now we can run inference on your model (note that inference is defined in the evaluate method in your model). Again, navigate to the project root (in a terminal) and run main.py via the following command (replace the config below with the one your defined for your model),

```
python main.py --mode inference --config_id podcnn_config
```

# Example: Using CMAMAE 
The following shows an example of extending the framework to support quality diversity (via pyribs). 

## Step 1: Define the config called configs/cmamae_config.py

```
CMAMAE_CONFIG = {
    "model":{
        "archive": {
            "solution_dim": 100,
            "dims": (100, 100),
            "max_bound": 100 / 2 * 5.12,
            "learning_rate": 0.01,
            "threshold_min": 0.0
        },
        "emitters": {
            "sigma": 0.5,
            "ranker": "imp",
            "selection_rule": "mu",
            "restart_rule": "basic",
            "batch_size": 36,
            "num_emitters": 15
        }},
    "data": {
        
    },
    "train": {
        "total_iterations": 10_000,
        "iters_per_progress_update": 500
    }
}
```

## Step 2: Add the config reference as a kehy-value pair to configs/__init__.py
Add the relative import to the top of the __init__.py file to import the config identifier. Then add the key 'cmamae_config'. This will be passed as an input argument to main.py.
```
from .cmamae_config import CMAMAE_CONFIG

CONFIGS_MAP = {
    ...other configs...
    "cmamae_config": CMAMAE_CONFIG
}
```


## Step 3: Define a model called CMAMAEModel in models/cmamae_model.py that inherits from BaseModel. 
```
import sys

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from .base_model import BaseModel


class CMAMAEModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.archive_config = self.config.model['archive']
        self.emitters_config = self.config.model['emitters']
        self.train_config = self.config.train
        self.build()
        

    def _build_archive(self):
        archive = GridArchive(solution_dim=self.archive_config["solution_dim"],
                      dims=self.archive_config["dims"],
                      ranges=[(-self.archive_config["max_bound"], self.archive_config["max_bound"]), (-self.archive_config["max_bound"], self.archive_config["max_bound"])],
                      learning_rate=self.archive_config["learning_rate"],
                      threshold_min=self.archive_config["threshold_min"])

        return archive

    def _build_result_archive(self):
        archive = GridArchive(solution_dim=self.archive_config["solution_dim"],
                      dims=self.archive_config["dims"],
                      ranges=[(-self.archive_config["max_bound"], self.archive_config["max_bound"]), (-self.archive_config["max_bound"], self.archive_config["max_bound"])],
                      )

        return archive


    def _build_emitters(self):
        emitters = [
            EvolutionStrategyEmitter(
                self.archive,
                x0=np.zeros(100),
                sigma0=self.emitters_config["sigma"],
                ranker=self.emitters_config["ranker"],
                selection_rule=self.emitters_config["selection_rule"],
                restart_rule=self.emitters_config["restart_rule"],
                batch_size=self.emitters_config["batch_size"],
            ) for _ in range(self.emitters_config["num_emitters"])
        ]

        return emitters

    def _build_scheduler(self, archive, emitters, result_archive):
        return Scheduler(archive, emitters, result_archive=result_archive)

    def load_data(self):
        pass

    def build(self):
        self.archive = self._build_archive()
        self.result_archive = self._build_result_archive()
        self.emitters = self._build_emitters()
        self.scheduler = self._build_scheduler(self.archive, self.emitters, self.result_archive)

    def sphere(self, solution_batch):
        """Sphere function evaluation and measures for a batch of solutions.

        Args:
            solution_batch (np.ndarray): (batch_size, dim) batch of solutions.
        Returns:
            objective_batch (np.ndarray): (batch_size,) batch of objectives.
            measures_batch (np.ndarray): (batch_size, 2) batch of measures.
        """
        dim = solution_batch.shape[1]

        # Shift the Sphere function so that the optimal value is at x_i = 2.048.
        sphere_shift = 5.12 * 0.4

        # Normalize the objective to the range [0, 100] where 100 is optimal.
        best_obj = 0.0
        worst_obj = (-5.12 - sphere_shift)**2 * dim
        raw_obj = np.sum(np.square(solution_batch - sphere_shift), axis=1)
        objective_batch = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

        # Calculate measures.
        clipped = solution_batch.copy()
        clip_mask = (clipped < -5.12) | (clipped > 5.12)
        clipped[clip_mask] = 5.12 / clipped[clip_mask]
        measures_batch = np.concatenate(
            (
                np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
                np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
            ),
            axis=1,
        )

        return objective_batch, measures_batch

    def train(self):
        total_itrs = self.train_config["total_iterations"]

        for itr in trange(1, total_itrs + 1, file=sys.stdout, desc='Iterations'):
            solution_batch = self.scheduler.ask()
            objective_batch, measure_batch = self.sphere(solution_batch)
            self.scheduler.tell(objective_batch, measure_batch)

            # Output progress every 500 iterations or on the final iteration.
            if itr % 500 == 0 or itr == total_itrs:
                tqdm.write(f"Iteration {itr:5d} | "
                        f"Archive Coverage: {self.result_archive.stats.coverage * 100:6.3f}%  "
                        f"Normalized QD Score: {self.result_archive.stats.norm_qd_score:6.3f}")

        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(self.result_archive, vmin=0, vmax=100)

    def evaluate(self):
        total_itrs = self.train_config["total_iterations"]

        for itr in trange(1, total_itrs + 1, file=self.train_config["file"], desc='Iterations'):
            solution_batch = self.scheduler.ask()
            objective_batch, measure_batch = self.sphere(solution_batch)
            self.scheduler.tell(objective_batch, measure_batch)

            # Output progress every 500 iterations or on the final iteration.
            if itr % 500 == 0 or itr == total_itrs:
                tqdm.write(f"Iteration {itr:5d} | "
                        f"Archive Coverage: {self.result_archive.stats.coverage * 100:6.3f}%  "
                        f"Normalized QD Score: {self.result_archive.stats.norm_qd_score:6.3f}")

        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(self.result_archive, vmin=0, vmax=100)
```

## Step 3: Add the model reference as a kehy-value pair to models/__init__.py
Add the relative import to the top of the __init__.py file to import the model identifier. Then add the key 'cmamae_model'. This will be passed as an input argument to main.py.
```
import sys,os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from .pod_cnn import PoDCNNModel
from .lego_model_3d import LegoModel3D
from .cmamae_model import CMAMAEModel # Added this line!

MODELS_MAP = {
    "podcnn_model": PoDCNNModel,
    "lego3d_model" : LegoModel3D,
    "cmamae_model": CMAMAEModel # Added this key-value pair
}
```

## Step 4: Run the model
Now to run the model make cd to the project root and run the following command,
```
python3 main.py --mode train --model cmamae_model --config cmamae_config
```
