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