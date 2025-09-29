import os
import csv
import copy
import numpy as np
import torch
from torch_geometric.data import Data
from dataclasses import dataclass, field
from typing import Any, List

from .utils import Normalizer

@dataclass
class Fitness:
    score: float = 0.0
    initial_components: List[float] = field(default_factory=list)
    components: List[float] = field(default_factory=list)


@dataclass
class MaterialProperties:
    name: str = "Steel"
    young_mods: np.ndarray = np.full((15,), 7e10)
    densities: np.ndarray = np.full((15,), 7872)
    poisson_ratios: np.ndarray = np.full((15,), 0.3)


@dataclass
class PhysicalState:
    forces: Any = None
    stresses: Any = None
    volumes: Any = None
    strains: Any = None
    strain_energies: Any = None


class Organism:
    def __init__(self, gen_id, pop_id, run_dir, seedling):
        self.generation_id = gen_id
        self.population_id = pop_id
        self.devo_step = 0
        self.run_dir = run_dir

        self.nodes = copy.deepcopy(seedling["nodes"])
        self.edges = copy.deepcopy(seedling["edges"])
        self.cs_areas = copy.deepcopy(seedling["cs_areas"])
        self.node_constraints = copy.deepcopy(seedling["node_constraints"])
        self.material = copy.deepcopy(seedling["materials"])
        self.physical_state = PhysicalState()

        # Normalizers for GNN inputs
        self.normalizer_node = Normalizer(2)
        self.normalizer_edge = Normalizer(2)

    def sense_environment(self, environment):
        """Sense environment to update physical state."""
        self.physical_state = environment.update_physical_state(self.nodes, self.edges, self.cs_areas,
                                                                self.physical_state, self.material)

    def get_cell_inputs(self, devo_step):
        # Input features for nodes are their coordinates
        N = self.nodes

        # Input features for edges are their strain energy and volume
        E = np.zeros((self.edges.shape[0], 2))
        E[:, 0] = self.physical_state.strain_energies
        E[:, 1] = self.physical_state.volumes

        # Observe and normalize the features
        self.normalizer_node.observe(np.mean(N, axis=0), np.var(N, axis=0))
        N_norm = self.normalizer_node.normalize(N)
        self.normalizer_edge.observe(np.mean(E, axis=0), np.var(E, axis=0))
        E_norm = self.normalizer_edge.normalize(E)

        return N_norm, E_norm, None, None, None

    def get_graph_data(self, device='cpu') -> Data:
        """
        Creates the PyTorch Geometric Data object for the GNN.
        """
        N_norm, E_norm, _, _, _ = self.get_cell_inputs(self.devo_step)

        nodes_tensor = torch.tensor(N_norm, dtype=torch.float32, device=device)
        edges_np = self.edges.astype("int64")
        edge_index = torch.tensor(edges_np.T, dtype=torch.long, device=device)
        edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.tensor(E_norm, dtype=torch.float32, device=device)
        edge_attr_undirected = torch.cat([edge_attr, edge_attr], dim=0)

        return Data(x=nodes_tensor, edge_index=edge_index_undirected, edge_attr=edge_attr_undirected)

    def update_with_cell_outputs(self, e_out, n_out, devo_step):
        """Update organism properties based on GNN outputs."""
        self.devo_step = devo_step
        self._update_node_coords(n_out)
        self._update_cs_areas(e_out)

    def _update_node_coords(self, coord_deltas):
        """Update node coordinates, respecting constraints."""
        coord_deltas = np.array(coord_deltas).reshape(-1, 2)
        coord_deltas[self.node_constraints, :] = 0.0
        self.nodes += coord_deltas

    def _update_cs_areas(self, area_deltas):
        """Update cross-sectional areas, ensuring they are non-negative."""
        area_deltas = np.array(area_deltas).reshape(-1)
        self.cs_areas += area_deltas
        self.cs_areas = np.maximum(self.cs_areas, 0.01)  # Enforce a minimum area

    def get_fitness(self, initial_components: List[float] = None) -> Fitness:
        """
        Returns a Fitness object.
        """
        total_strain_energy = np.sum(self.physical_state.strain_energies)
        total_volume = np.sum(self.physical_state.volumes)

        if initial_components is None:
            initial_components = [total_strain_energy, total_volume]

        se_norm = total_strain_energy / (initial_components[0] + 1e-8)
        vol_norm = total_volume / (initial_components[1] + 1e-8)

        current_score = se_norm + vol_norm
        current_components = [total_strain_energy, total_volume]

        return Fitness(
            score=current_score,
            initial_components=initial_components,
            components=current_components
        )

    def save_organism(self):
        """
        Saving the state of every organism at every step .
        This should only be enabled for debugging a single run, not for training.
        """
        # nodes_path = os.path.join(self.run_dir, "nodes.csv")
        # self._write_csv(nodes_path, self.nodes)
        pass  # Disabled for performance

    def _write_csv(self, file_path, data):
        # with open(file_path, 'a', newline='') as csv_file:
        #     writer = csv.writer(csv_file)
        #     row_data = np.concatenate(([self.generation_id, self.population_id, self.devo_step], data.flatten()))
        #     writer.writerow(row_data)
        pass  # Disabled for performance