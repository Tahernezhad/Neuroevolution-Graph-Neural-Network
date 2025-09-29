import torch
from .organism import Organism, Fitness
from .gnn_model import GATModel, GCNModel, SAGEModel


class NeuralNet:
    def __init__(self, model_config: dict, device: str = "cpu", custom_init_limits=None):
        self.device = torch.device(device)
        self.model_config = model_config

        model_name = model_config.get("model", "gcn").lower()
        hidden = int(model_config.get("hidden", 64))
        layers = int(model_config.get("layers", 3))
        heads = int(model_config.get("heads", 4))
        dropout = float(model_config.get("dropout", 0.1))

        if model_name == "gat":
            self.model = GATModel(hidden_dim=hidden, heads=heads, dropout=dropout)
        elif model_name == "gcn":
            self.model = GCNModel(hidden_dim=hidden, num_layers=layers, dropout=dropout)
        else:  # sage
            self.model = SAGEModel(hidden_dim=hidden, num_layers=layers, dropout=dropout)

        self.model.to(self.device)

        if custom_init_limits is not None:
            self._initialize_weights_uniformly(*custom_init_limits)

    def _initialize_weights_uniformly(self, min_val: float, max_val: float):
        with torch.no_grad():
            for param in self.model.parameters():
                param.data.uniform_(min_val, max_val)

    @torch.no_grad()
    def evaluate(self, seedling, environment, max_devo_steps, run_dir, gen_id, pop_id):
        self.model.eval()
        org = Organism(gen_id, pop_id, run_dir, seedling)
        org.sense_environment(environment)

        organism_id_str = f"{gen_id}-{pop_id}"
        initial_fitness: Fitness = org.get_fitness()
        current_fitness_score = initial_fitness.score

        for step in range(1, max_devo_steps + 1):
            graph_data = org.get_graph_data(device=self.device)
            e_pred_dir, n_pred = self.model(graph_data)

            E = org.edges.shape[0]
            e_pred_undirected = e_pred_dir[:E].cpu().numpy()
            n_pred_np = n_pred.cpu().numpy()
            org.update_with_cell_outputs(e_pred_undirected, n_pred_np, step)
            org.sense_environment(environment)

            current_fitness: Fitness = org.get_fitness(initial_fitness.components)
            current_fitness_score = current_fitness.score

        return -current_fitness_score, organism_id_str