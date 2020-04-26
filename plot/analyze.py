import math
import os 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import *

sns.set()

class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (((df["radius"] >= radius) | (df["radius"] == -1))).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))


class EmpiricalAccuracy(Accuracy):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def at_radii(self, radii: np.ndarray, attack: str, method: str) -> np.ndarray:
        accuracies = []
        for radius in radii:
            file_path = os.path.join(self.data_dir, '{}_{:.3f}/{}/predictions'.format(attack, radius, method))
            df = pd.read_csv(file_path, delimiter="\t")
            accuracies.append(self.at_radius(df, radius))
        return np.array(accuracies)

    def at_radius(self, df: pd.DataFrame, radius: float):
        return df["correct"].mean()

class Line(object):
    def __init__(self, quantity: Accuracy, legend: str = None, plot_fmt: str = "", scale_x: float = 1, alpha: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x
        self.alpha = alpha

def plot_empirical_accuracy_per_sigma_against_original_one_sample(outfile: str, title: str, max_radius: float,
                            methods: List[Line], methods_original: List[Line], radius_step: float = 0.01, upper_bounds=False, xlabel="$\ell_2$ radius") -> None:

    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure(figsize=(8, 4), dpi=80)
    color = ['b', 'orange', 'g', 'r', 'purple', 'pink', 'cyan', 'black']
    for it, line in enumerate(methods):
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), color[it], alpha=line.alpha, label=line.legend)

    for it, line in enumerate(methods_original):
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), color[it], dashes=[2, 2], alpha=line.alpha, label=line.legend)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Empirical Accuracy", fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()