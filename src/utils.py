import os
import numpy as np

from enum import Enum
from datetime import datetime


def make_run_dir():
    """Makes new directory for run.

    Returns:
        Run directory path.
    """
    project_dir_path = os.getcwd()
    if project_dir_path.endswith("src"):
        project_dir_path = project_dir_path[:-len("src")]

    data_dir_path = os.path.join(project_dir_path, "data")
    if not os.path.isdir(data_dir_path):
        os.mkdir(data_dir_path)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    run_dir_path = os.path.join(data_dir_path, dt_string)
    if not os.path.isdir(run_dir_path):
        os.mkdir(run_dir_path)

    return run_dir_path


class Normalizer():
    def __init__(self, input_space):
        self.mean = np.zeros(input_space)
        self.var = np.zeros(input_space)
        self.count = 1e-4
        self.batch_count = 1
        self.epsilon = 1e-8
        self.clipob = 10.0

    def observe(self, org_mean, org_var):
        delta = org_mean - self.mean
        tot_count = self.count + self.batch_count

        self.mean = self.mean + delta * self.batch_count / tot_count
        m_a = self.var * self.count
        m_b = org_var * self.batch_count
        m = m_a + m_b + np.square(delta) * self.count * self.batch_count / tot_count
        self.var = m / tot_count
        self.count = tot_count

    def normalize(self, inputs):
        return np.clip((inputs - self.mean) / np.sqrt(self.var + self.epsilon), -self.clipob, self.clipob)
    

class FitCriteria(Enum):
    """Fitness criteria available in the system.
    
    Attributes:
        total_strain_energy: The sum of the strain energy values for all members in the organism.
        total_volume: The sum of the volume values for all members in the organism.
    """
    total_strain_energy = 0
    total_volume = 1


class NetworkInputs(Enum):
    volume = 0
    strain_energy = 1
    point_coord = 2


def num_network_inputs(network_inputs):
    num_inputs = 0

    if NetworkInputs.volume in network_inputs:
        num_inputs += 1
    if NetworkInputs.strain_energy in network_inputs:
        num_inputs += 1
    if NetworkInputs.point_coord in network_inputs:
        num_inputs += 2

    return num_inputs
