import numpy as np
from time import time

from src.utils import make_run_dir
from src.organism import MaterialProperties
from src.environment import Environment
from src.genetic_algorithm import GeneticAlgorithm


def define_seedling():
    """Produces the seedling parameters."""
    nodes = np.array([[0.0, 0.0],
                      [12.5, 21.650635],
                      [25.0, 0.0],
                      [37.5, 21.650635],
                      [50.0, 0.0],
                      [62.5, 21.650635],
                      [75.0, 0.0],
                      [87.5, 21.650635],
                      [100.0, 0.0]])

    edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5], [4, 6], [5, 6],
                      [5, 7], [6, 7], [6, 8], [7, 8]])

    cs_areas = np.full((edges.shape[0],), 1.0)
    node_constraints = np.array([0, 2, 4, 6, 8])

    materials = MaterialProperties()
    return {"nodes": nodes, "edges": edges, "cs_areas": cs_areas, "materials": materials,
            "node_constraints": node_constraints}


def define_environment():
    """Defines the environmental conditions."""
    reactions = np.array([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]])
    loads = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, -17000], [0, 0], [0, 0], [0, 0], [0, 0]])
    return Environment(reactions=reactions, loads=loads)


def main():
    # --- Model/Network parameters
    model_name = "gcn"
    hidden_dim = 120
    num_layers = 2
    dropout_rate = 0.1

    WEIGHT_MIN = -5.0
    WEIGHT_MAX = 5.0

    num_generations = 20
    population_size = 100
    num_devo_steps = 10

    crossover_prob = 0.9
    crossover_eta = 20.0

    mutation_prob = 0.1
    mutation_eta = 20.0

    random_seed = 2

    np.random.seed(random_seed)
    print(f"Random seed: {random_seed}")

    model_config = {
        "model": model_name, "hidden": hidden_dim,
        "layers": num_layers, "heads": 4, "dropout": dropout_rate,
    }

    seedling = define_seedling()
    environment = define_environment()
    run_dir = make_run_dir()
    start_time = time()

    ga = GeneticAlgorithm(
        seedling=seedling,
        environment=environment,
        model_config=model_config,
        generations=num_generations,
        population_size=population_size,
        run_dir=run_dir,
        verbose=True,
        print_every=1,
        num_devo_steps=num_devo_steps,
        weight_init_limits=(WEIGHT_MIN, WEIGHT_MAX),
        crossover_prob=crossover_prob,
        crossover_eta=crossover_eta,
        mutation_prob=mutation_prob,
        mutation_eta=mutation_eta
    )

    ga.fit()
    print(f"Finished in {round((time() - start_time) / 60.0, 3)} minutes")


if __name__ == "__main__":
    main()