import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch

from src.environment import Environment
from src.organism import Organism, MaterialProperties, Fitness # [MODIFIED]
from src.network import NeuralNet


def define_seedling():
    """Defines the initial truss structure (same as in main.py)."""
    nodes = np.array([[0.0, 0.0], [12.5, 21.650635], [25.0, 0.0],
                      [37.5, 21.650635], [50.0, 0.0], [62.5, 21.650635],
                      [75.0, 0.0], [87.5, 21.650635], [100.0, 0.0]])
    edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4],
                      [3, 5], [4, 5], [4, 6], [5, 6], [5, 7], [6, 7],
                      [6, 8], [7, 8]])
    cs_areas = np.full((edges.shape[0],), 1.0)
    node_constraints = np.array([0, 2, 4, 6, 8])
    materials = MaterialProperties()
    return {"nodes": nodes, "edges": edges, "cs_areas": cs_areas,
            "materials": materials, "node_constraints": node_constraints}


def define_environment():
    """Defines the physical loads and reactions (same as in main.py)."""
    reactions = np.array([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                          [0, 0], [0, 0], [0, 1]])
    loads = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, -17000], [0, 0],
                      [0, 0], [0, 0], [0, 0]])
    return Environment(reactions=reactions, loads=loads)


def plot_truss(ax, nodes, edges, cs_areas, title=""):
    """Helper function to draw the truss structure."""
    for i, (n1, n2) in enumerate(edges):
        ax.plot([nodes[n1, 0], nodes[n2, 0]],
                [nodes[n1, 1], nodes[n2, 1]],
                lw=max(cs_areas[i] * 2, 0.1), color="blue", zorder=2)
    ax.scatter(nodes[:, 0], nodes[:, 1], color="red", zorder=5)
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.grid(True)


@torch.no_grad()
def replay_best_gnn(run_dir: str, devo_steps: int):
    """
    Loads the best GNN from a run and replays its developmental process,
    generating visualizations and data logs.
    """
    # 1. Load the best network from the specified run directory
    best_pkl = os.path.join(run_dir, "best_network.pkl")
    if not os.path.exists(best_pkl):
        print(f"[ERROR] {best_pkl} not found. Did you run the training script first?");
        return

    best_network: NeuralNet = pickle.load(open(best_pkl, "rb"))
    best_network.model.eval()

    # 2. Set up directories for saving results
    results_dir = os.path.join(run_dir, "results")
    frames_dir = os.path.join(results_dir, "best_devo_frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    # 3. Initialize the environment and organism
    env = define_environment()
    seed = define_seedling()

    org = Organism(gen_id=-1, pop_id=-1, run_dir=run_dir, seedling=seed)
    org.sense_environment(env)

    # Get initial fitness state for normalization
    initial_fitness: Fitness = org.get_fitness()

    # 4. Set up history tracking
    n_nodes = org.nodes.shape[0]
    n_edges = org.edges.shape[0]
    node_hist = np.zeros((devo_steps + 1, n_nodes, 2))
    area_hist = np.zeros((devo_steps + 1, n_edges))
    node_hist[0, :, :] = org.nodes
    area_hist[0, :] = org.cs_areas

    # History for plotting fitness components
    E_hist, V_hist, C_hist = [], [], []

    # 5. Create and save the initial frame
    frame_paths = []

    def save_frame(step: int, caption: str):
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_truss(ax, org.nodes, org.edges, org.cs_areas, caption)
        p = os.path.join(frames_dir, f"frame_{step:03d}.png")
        fig.savefig(p, dpi=100)
        plt.close(fig)
        frame_paths.append(p)

    save_frame(0, "Step 0 (Seedling)")

    # 6. Main Developmental Loop
    print("Starting developmental replay...")
    for step in range(1, devo_steps + 1):

        e_pred, n_pred = best_network.model(org.get_graph_data(device='cpu'))
        e_pred_undirected = e_pred[:n_edges].cpu().numpy()
        n_pred_np = n_pred.cpu().numpy()

        org.update_with_cell_outputs(e_pred_undirected, n_pred_np, step)
        org.sense_environment(env)

        current_fitness_state: Fitness = org.get_fitness(initial_fitness.components)
        sv = current_fitness_state.components
        E_hist.append(sv[0])
        V_hist.append(sv[1])
        C_hist.append(current_fitness_state.score)

        node_hist[step, :, :] = org.nodes
        area_hist[step, :] = org.cs_areas

        caption = f"Step {step} | E={sv[0]:.3e}, V={sv[1]:.2f}, Cost={current_fitness_state.score:.3f}"
        save_frame(step, caption)
        print(f"  ... Step {step} complete.")

    # 7. Create GIF from frames
    if frame_paths:
        gif_path = os.path.join(results_dir, "best_devo.gif")
        imageio.mimsave(gif_path, [imageio.imread(p) for p in frame_paths], fps=2, loop=0)
        print(f"Saved developmental GIF -> {gif_path}")

    # 8. Save data to CSV files
    header_nodes = ["step"] + [f"x{i}" for i in range(n_nodes)] + [f"y{i}" for i in range(n_nodes)]
    with open(os.path.join(results_dir, "node_positions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header_nodes)
        for s in range(devo_steps + 1):
            row = [s] + node_hist[s, :, 0].tolist() + node_hist[s, :, 1].tolist()
            w.writerow(row)

    with open(os.path.join(results_dir, "edge_areas.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"edge{e}" for e in range(n_edges)])
        for s in range(devo_steps + 1):
            w.writerow([s] + area_hist[s, :].tolist())
    print("Saved node position and edge area data to CSVs.")

    # 9. Create and save plots
    steps_axis = np.arange(1, devo_steps + 1)
    fig, axs = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    plots_data = [
        (axs[0], E_hist, "Strain Energy", "blue"),
        (axs[1], V_hist, "Volume", "green"),
        (axs[2], C_hist, "Total Cost (Normalized Fitness)", "red")
    ]
    for ax, data, label, color in plots_data:
        ax.plot(steps_axis, data, 'o-', color=color)
        ax.set_title(label)
        ax.grid(True)
    axs[2].set_xlabel("Developmental Step")
    fig.tight_layout()
    plot_path = os.path.join(results_dir, "gnn_devo_plot.jpg")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved fitness trajectory plot -> {plot_path}")


def plot_generation_rewards(run_dir: str):
    """Plots the best and average rewards per generation from training."""
    csv_path = os.path.join(run_dir, "reward_plot.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} missing; skipping reward plot.");
        return

    gens, best, avg = [], [], []
    with open(csv_path) as f:
        r = csv.reader(f);
        next(r)  # Skip header
        for g, b, a in r:
            gens.append(int(g));
            best.append(float(b));
            avg.append(float(a))

    plt.figure(figsize=(8, 5))
    plt.plot(gens, best, 'o-', label="Best Reward per Gen")
    plt.plot(gens, avg, 'x-', label="Average Reward per Gen", alpha=0.7)
    plt.xlabel("Generation")
    plt.ylabel("Reward (Negative Fitness)")
    plt.title("Evolutionary Training Progress")
    plt.grid(True);
    plt.legend()

    out_path = os.path.join(run_dir, "results", "gnn_evo_plot.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved reward trajectory plot -> {out_path}")


# ──────────────────────────── main ──────────────────────────────────
def main():

    # For example: 'data/29-09-2025-01-54-00'
    run_dir = "data/29-09-2025-16-30-55"
    devo_steps = 10

    if not os.path.isdir(run_dir):
        print(f"\n[ERROR] The specified run directory does not exist: '{run_dir}'")
        print("Please make sure you have run the main.py training script first,")
        print("and that the path is correct.\n")
        return

    replay_best_gnn(run_dir, devo_steps)
    plot_generation_rewards(run_dir)


if __name__ == "__main__":
    main()
