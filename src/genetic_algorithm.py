import numpy as np
import csv
import os
import pickle
import random
import torch

from .network import NeuralNet
from .ea_utils import simulated_binary_crossover, polynomial_mutation
from .organism import Fitness


class GeneticAlgorithm():
    def __init__(self, seedling, environment, model_config, generations=500, population_size=64,
                 run_dir=None, verbose=False, print_every=1, num_devo_steps=10,
                 weight_init_limits=(-1.0, 1.0),
                 crossover_prob=0.9, crossover_eta=20.0,
                 mutation_prob=0.1, mutation_eta=20.0):

        self.generations = generations
        self.population_size = population_size
        self.verbose = verbose
        self.print_every = print_every
        self.run_dir = run_dir
        self.seedling = seedling
        self.environment = environment
        self.num_devo_steps = num_devo_steps
        self.model_config = model_config
        self.weight_bounds = weight_init_limits

        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_prob = mutation_prob
        self.mutation_eta = mutation_eta

        self.best_network = None
        self.best_reward = -np.inf
        self.best_organism_id = None

        print("Creating initial population...")
        self.networks = [
            NeuralNet(model_config=self.model_config, custom_init_limits=self.weight_bounds)
            for _ in range(self.population_size)
        ]

    def fit(self):
        stats_gens, stats_highest, stats_avg = [], [], []

        print("Evaluating initial population...")
        initial_eval_results = [
            net.evaluate(self.seedling, self.environment, self.num_devo_steps, self.run_dir, 0, i)
            for i, net in enumerate(self.networks)
        ]
        rewards = np.array([res[0] for res in initial_eval_results])
        organism_ids = [res[1] for res in initial_eval_results]

        for gen in range(self.generations):
            offspring = self._create_offspring(self.networks, rewards)

            combined_population = self.networks + offspring

            offspring_eval_results = [
                net.evaluate(self.seedling, self.environment, self.num_devo_steps, self.run_dir, gen,
                             i + self.population_size)
                for i, net in enumerate(offspring)
            ]
            offspring_rewards = np.array([res[0] for res in offspring_eval_results])
            offspring_ids = [res[1] for res in offspring_eval_results]

            combined_rewards = np.concatenate([rewards, offspring_rewards])
            combined_ids = organism_ids + offspring_ids

            survivor_indices = np.argsort(-combined_rewards)[:self.population_size]

            self.networks = [combined_population[i] for i in survivor_indices]
            rewards = combined_rewards[survivor_indices]
            organism_ids = [combined_ids[i] for i in survivor_indices]

            best_gen_idx = np.argmax(rewards)
            best_gen_reward = rewards[best_gen_idx]

            if best_gen_reward > self.best_reward:
                self.best_reward = best_gen_reward
                self.best_network = self.networks[best_gen_idx]
                self.best_organism_id = organism_ids[best_gen_idx]

            if self.verbose and (gen % self.print_every == 0):
                print(
                    f'Gen: {gen} | Best Reward: {best_gen_reward:.6f} | Avg Reward: {rewards.mean():.6f} | '
                    f'Best Overall: {self.best_reward:.6f} (Org {self.best_organism_id})'
                )

            stats_gens.append(gen)
            stats_highest.append(best_gen_reward)
            stats_avg.append(rewards.mean())

        self._write_reward_plot_csv(stats_gens, stats_highest, stats_avg)
        self.save_best_network(self.best_network)
        print(f"\nFinished. Best organism: {self.best_organism_id} with reward {self.best_reward:.6f}")

    def _create_offspring(self, parents: list, parent_rewards: np.ndarray) -> list:
        offspring = []
        while len(offspring) < self.population_size:
            p1 = self._tournament_selection(parents, parent_rewards, k=3)
            p2 = self._tournament_selection(parents, parent_rewards, k=3)

            c1, c2 = simulated_binary_crossover(p1, p2, self.crossover_eta, self.crossover_prob, self.model_config, self.weight_bounds)
            polynomial_mutation(c1, self.mutation_eta, self.mutation_prob, self.weight_bounds)
            polynomial_mutation(c2, self.mutation_eta, self.mutation_prob, self.weight_bounds)

            offspring.extend([c1, c2])
        return offspring[:self.population_size]

    def _tournament_selection(self, population: list, rewards: np.ndarray, k: int) -> NeuralNet:
        indices = random.sample(range(len(population)), k)

        best_index_in_tournament = -1
        best_reward = -np.inf
        for i in indices:
            if rewards[i] > best_reward:
                best_reward = rewards[i]
                best_index_in_tournament = i

        return population[best_index_in_tournament]

    def _write_reward_plot_csv(self, gens, highest_rewards, avg_rewards):
        reward_plot_path = os.path.join(self.run_dir, "reward_plot.csv")
        with open(reward_plot_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Generation", "Best Reward", "Avg Reward"])
            for i in range(len(gens)):
                writer.writerow([gens[i], highest_rewards[i], avg_rewards[i]])

    def save_best_network(self, network):
        if network is None:
            print("Warning: No best network was found to save.")
            return
        network_path = os.path.join(self.run_dir, "best_network.pkl")
        with open(network_path, 'wb') as handle:
            pickle.dump(network, handle, protocol=pickle.HIGHEST_PROTOCOL)