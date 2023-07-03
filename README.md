# legopcg

This project is a lego building environment for analyzing and experimenting with various generative learning algorithms.


 <img src="media/pod_repair_1.gif" width="450"/>

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
import os
import pathlib
import datetime


basepath = pathlib.Path(os.getcwd()).parents[0]
PODCNN_TRAINER_CONFIG = {
    "data": {
        "path": f"{basepath}/legopcg/data/input/podcnn/vintage_car",
        "goals_path": f"{basepath}/legopcg/data/goals/vintage_car_1.mpd",
        "output_path": f"{basepath}/legopcg/data/output",
        "action_dim": 37,
        "train_data_path": f"{basepath}/legopcg/data/trajectories/racers",
        "obs_size": 6,
        "use_signed_inputs": True,
    },
    "train": {
        "batch_size": 64,
        "buffer_size": 1000,
        "epochs": 500,
        "steps_per_epoch": 128,
        "val_subsplits": 5,
        "optimizer": {"type": "adam"},
        "metrics": ["accuracy"],
    },
    "model": {
        "obs_size": 6,
        "action_dim": 37,
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3,
        },
        "output": 3,
        "model_save_path": f"{basepath}/legopcg/saved_models",
        "model_name": f"pod_cnn_{datetime.datetime.now()}.h5",
    },
}
```

Next, register the config by accessing configs/__init__.py and adding the following code,

```
import sys,os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# add your config import here after the other config imports
from .podcnn_trainer_config import PODCNN_TRAINER_CONFIG

CONFIGS_MAP = {
    ...
    "podcnn_trainer_example": PODCNN_TRAINER_CONFIG,
    ...
}
```


## Step 3: Defining a model
All models inherit from the BaseModel class in model/base_model.py. Therefore, your model must implement the following methods

```
class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self):
        pass

    @abstractmethod
    def get_trainer_id(self):
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raise NotImplementedError
```


Here is an example convolutional neural network Path of Destruction model,

```

from .base_model import BaseModel


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

        x = Concatenate()([convolved_features, x_lego_blocks])
        x = Dense(128)(x)

        output = [
            Dense(action_dim, activation="softmax")(x),
        ]
        conditional_cnn_model = Model(inputs, output)

        return conditional_cnn_model
```


## Step 4: Defining a dataloader
Per your implementation requirements, a dataloader may be required (e.g. for supervised learning problems). All datraloaders should inherit from the base class below.
```
class BaseDataLoader(ABC):
    def __init__():
        pass

    @abstractmethod
    def get_trainer_id(self):
        raise NotImplementedError

    @abstractmethod
    def load_data(self):
        raise NotImplementedError
```

Below is an example dataloader that implements the abstract methods get_trainer_id and load_data, respectively.

```
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
        signed_output = np.column_stack((num_lego_pieces_signed,))

        action_dim = 37
        obs_size = 6

        for idx in range(len(df)):
            x = df.iloc[idx, :].values.astype("int32")
            row = []
            for val in x:
                oh = [0] * 37
                oh[val] = 1
                row.append(oh)
            X.append(np.array(row).reshape((obs_size, obs_size, obs_size, action_dim)))

        X = np.array(X)

        return [
            K.constant(X),
            K.constant(np.array(cond_input_target)),
            K.constant(np.array(signed_output)),
        ], y

```

Finally, to register the dataloader reference it in dataloaders/__init__.py like below.

```
from .podcnn_dataloader import PoDCNNDataLoader

DATALOADERS_MAP = {
    ...
    PoDCNNDataLoader.get_trainer_id(): PoDCNNDataLoader
    ...
}
```




## Step 5: Define a Trainer
Define a trainer that handles training a model via loading in data using a dataloader you define (if applicable). All Trainers inherit from the following base class.

```
class BaseTrainer(ABC):
    """Abstract Trainer class that is inherited to all models"""

    def __init__(self, config_id):
        self.config = Config.from_json(configs.CONFIGS_MAP[config_id])

    @abstractmethod
    def get_id(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError
```

Below is an example dataloader that implements the abstract methods.

```
class PoDCNNTrainer(BaseTrainer):
    def __init__(self, config_id):
        super().__init__(config_id)
        self.model = MODELS_MAP[PoDCNNModel.get_trainer_id()]().build(
            self.config
        )
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
```

Lastly, register the your trainer by referencing it in trainers/__init__.py.

```
from .podcnn_trainer import PoDCNNTrainer

TRAINERS_MAP = {
    PoDCNNTrainer.get_id(): PoDCNNTrainer
}
```

## Step 6: Run training
To call a trainer's train method navigate to the project root (in a terminal) and run train.py via the following command,

```
python train.py --config_id podcnn_trainer_config
```

## Step 7: Run inference
That's it! Now we can run inference on your model (note that inference is defined in the evaluate method in your model). Again, navigate to the project root (in a terminal) and run main.py via the following command (replace the config below with the one your defined for your model),

```
python inference.py --config_id podcnn_inference_config
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

## Step 3: Add the model reference as a key-value pair to models/__init__.py
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

This paper should be cited if legopcg-related code is used from this project in any way.

```
@misc{siper2023controllable,
      title={Controllable Path of Destruction}, 
      author={Matthew Siper and Sam Earle and Zehua Jiang and Ahmed Khalifa and Julian Togelius},
      year={2023},
      eprint={2305.18553},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
