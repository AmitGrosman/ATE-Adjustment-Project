import os
import time
import random
import logging
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Configuration and Setup

# Configure logging to record experiment details and debug information
logging.basicConfig(
    filename='experiment.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def log_and_print(message, level='info'):
    """Logs messages and prints them to the console."""
    getattr(logging, level)(message)
    print(message)


# Dataset Creation
def create_dataset(size=50, loc_1=70, loc_0=40, num_confounders=2, confounder_values_per_W=3):
    """Generates a dataset with binary treatment and multiple confounders (W1, W2, ...)."""
    T = np.random.randint(0, 2, size)
    O = np.where(
        T == 1,
        np.random.normal(loc=loc_1, scale=20, size=size),
        np.random.normal(loc=loc_0, scale=20, size=size)
    )
    O = np.clip(O, 0, 100)

    confounders = {f'W{i+1}': np.random.randint(0, confounder_values_per_W, size) for i in range(num_confounders)}

    return pd.DataFrame({'T': T, 'O': O, **confounders})


# ATE Calculation Functions

def calc_initial_aggregates(dataset, confounders):
    """Calculates sums and counts for each subgroup with multiple confounders."""
    aggregates = {}
    T_values = dataset['T'].unique()

    # Get all unique combinations of confounders (cartesian product)
    confounder_combinations = dataset.groupby(confounders).size().index

    for T in T_values:
        for combo in confounder_combinations:
            subgroup = dataset[(dataset['T'] == T) & (dataset[confounders] == combo).all(axis=1)]
            aggregates[(T, combo)] = {'sum_O': subgroup['O'].sum(), 'count': len(subgroup)}

    return aggregates



def calc_ATE_incremental(aggregates, confounder_combinations):
    """Calculates the ATE based on aggregates for multiple confounders."""
    E_Y_T1 = {}
    E_Y_T0 = {}

    for combo in confounder_combinations:
        key_T1 = (1, combo)  # key for T=1 and confounder combination
        key_T0 = (0, combo)  # key for T=0 and confounder combination

        # Calculate the means for T=1 and T=0 for each combination
        E_Y_T1[combo] = (aggregates[key_T1]['sum_O'] / aggregates[key_T1]['count']) if key_T1 in aggregates and aggregates[key_T1]['count'] > 0 else 0
        E_Y_T0[combo] = (aggregates[key_T0]['sum_O'] / aggregates[key_T0]['count']) if key_T0 in aggregates and aggregates[key_T0]['count'] > 0 else 0

    # Calculate the weighted sum of E_Y_T1 and E_Y_T0
    total_count = sum([aggregates.get((1, combo), {'count': 0})['count'] + aggregates.get((0, combo), {'count': 0})['count'] for combo in confounder_combinations])
    p_combo = {combo: (aggregates.get((1, combo), {'count': 0})['count'] + aggregates.get((0, combo), {'count': 0})['count']) / total_count if total_count > 0 else 0
               for combo in confounder_combinations}

    return sum(p_combo[combo] * (E_Y_T1[combo] - E_Y_T0[combo]) for combo in confounder_combinations)



def calc_ATE_with_mask(aggregates, dataset, mask, confounder_combinations, confounders):
    """Recalculates ATE with the mask applied for multiple confounders."""
    E_Y_T1 = {combo: dataset['O'][(dataset['T'] == 1) & (dataset[confounders] == combo).all(axis=1) & mask].mean() if np.sum(
        (dataset['T'] == 1) & (dataset[confounders] == combo).all(axis=1) & mask) > 0 else 0
              for combo in confounder_combinations}
    E_Y_T0 = {combo: dataset['O'][(dataset['T'] == 0) & (dataset[confounders] == combo).all(axis=1) & mask].mean() if np.sum(
        (dataset['T'] == 0) & (dataset[confounders] == combo).all(axis=1) & mask) > 0 else 0
              for combo in confounder_combinations}

    total_count = np.sum(mask)
    p_combo = {combo: np.sum(((dataset['T'] == 1) | (dataset['T'] == 0)) & (dataset[confounders] == combo).all(axis=1) & mask) / total_count
               if total_count > 0 else 0 for combo in confounder_combinations}

    return sum(p_combo[combo] * (E_Y_T1[combo] - E_Y_T0[combo]) for combo in confounder_combinations)


def ATE_after_row_removal(row_idx, dataset, mask, aggregates, confounder_combinations, confounders):
    """Calculates ATE as if the row was removed."""
    temp_mask = mask.copy()
    temp_mask[row_idx] = False
    return calc_ATE_with_mask(aggregates, dataset, temp_mask, confounder_combinations, confounders)


def calculate_row_impact_and_sort(dataset, mask, aggregates, target_ATE, confounder_combinations, confounders):
    """Helper to calculate and sort rows by their impact on ATE."""
    row_diffs = []
    for i in range(len(dataset)):
        if mask[i]:
            new_ATE = ATE_after_row_removal(i, dataset, mask, aggregates, confounder_combinations, confounders)
            diff = abs(new_ATE - target_ATE)
            row_diffs.append((i, diff))
    return sorted(row_diffs, key=lambda x: x[1])


def per_group_removal_algorithm(dataset_info, target_ATE, epsilon, confounder_combinations, confounders):
    """Algorithm to remove rows per group (W1, W2, ...) and select the one affecting ATE the most for multi-valued confounders."""
    dataset, aggregates = dataset_info
    mask = np.ones(len(dataset), dtype=bool)
    total_removals = 0
    removals_by_group = {f'(T={T}, {combo})': 0 for T in [0, 1] for combo in confounder_combinations}
    iterations, differences = [], []

    sorted_indices = {T: {combo: np.where((dataset['T'] == T) & (dataset[confounders] == combo).all(axis=1))[0][np.argsort(dataset['O'][(dataset['T'] == T) & (dataset[confounders] == combo).all(axis=1)])]
                          for combo in confounder_combinations} for T in [0, 1]}

    current_ATE = calc_ATE_with_mask(aggregates, dataset, mask, confounder_combinations, confounders)

    while True:
        best_diff, row_to_remove, group_to_remove = float('inf'), None, None

        # Check ATE and decide whether to remove highest or lowest from each group
        if current_ATE > target_ATE + epsilon:
            # ATE is above target, remove highest from each group
            candidates = [(T, combo, sorted_indices[T][combo][mask[sorted_indices[T][combo]]][-1]) for T in [0, 1] for combo in confounder_combinations if np.any(mask[sorted_indices[T][combo]])]
        else:
            # ATE is below target, remove lowest from each group
            candidates = [(T, combo, sorted_indices[T][combo][mask[sorted_indices[T][combo]]][0]) for T in [0, 1] for combo in confounder_combinations if np.any(mask[sorted_indices[T][combo]])]

        # Simulate removing the selected rows from each group and find the best one
        for T, combo, candidate in candidates:
            new_ATE = ATE_after_row_removal(candidate, dataset, mask, aggregates, confounder_combinations, confounders)
            diff = abs(new_ATE - target_ATE)
            if diff < best_diff:
                best_diff = diff
                row_to_remove = candidate
                group_to_remove = (T, combo)

        # Stop if no rows to remove
        if row_to_remove is None:
            break

        # Update mask to remove the chosen row
        mask[row_to_remove] = False
        total_removals += 1
        removals_by_group[f'(T={group_to_remove[0]}, {group_to_remove[1]})'] += 1

        # Update current ATE
        current_ATE = calc_ATE_with_mask(aggregates, dataset, mask, confounder_combinations, confounders)
        iterations.append(total_removals)
        differences.append(current_ATE)

        # Check if the target ATE has been reached within the epsilon range
        if target_ATE - epsilon <= current_ATE <= target_ATE + epsilon:
            break

    # Ensure the function always returns a valid tuple
    return total_removals, removals_by_group, iterations, differences


# Plotting Functions

def plot_average_metric(metric, ylabel, size_results, sizes, epsilon_adjust_factors, graphs_folder, current_size_idx):
    """Plots the average metric against dataset sizes for all algorithms."""
    sns.set_style("whitegrid") # Set Seaborn style

    # Create a folder for the specific metric
    metric_folder = os.path.join(graphs_folder, f"{metric}_plots")
    os.makedirs(metric_folder, exist_ok=True)

    # Loop through each epsilon factor and create a plot
    for factor_idx, epsilon_factor in enumerate(epsilon_adjust_factors):
        plt.figure(figsize=(12, 8))  # New figure for each epsilon factor

        for algo_name in size_results:
            metrics = size_results[algo_name]

            # Compute average metric
            avg_metric = []
            for i in range(current_size_idx + 1):
                size = sizes[i]
                values = metrics[metric][factor_idx][size].get('values', [])
                if values:
                    avg = np.mean(values)
                else:
                    avg = float('nan')
                avg_metric.append(avg)

            # Plot the metric
            plt.plot(sizes[:current_size_idx + 1], avg_metric, marker='o', label=f'{algo_name} {ylabel}')

        # Configure plot
        plt.title(f'Average {ylabel} vs Dataset Size (Epsilon Factor: {epsilon_factor})', fontsize=16)
        plt.xlabel('Dataset Size', fontsize=14)
        plt.ylabel(f'Average {ylabel}', fontsize=14)
        plt.legend(title='Algorithms')
        plt.ylim(bottom=0)  # Y-axis starts at 0
        plt.grid(True, which='both', axis='both')  # Full grid
        plt.xticks(sizes[:current_size_idx + 1])
        plt.tight_layout()

        # Save the plot
        plot_filename = f"average_{metric}_vs_size_epsilon_{epsilon_factor}.png"
        plt.savefig(os.path.join(metric_folder, plot_filename))
        plt.close()

def plot_experiment_results(size_results, sizes, epsilon_adjust_factors, graphs_folder, current_size_idx):
    """Plots all experiment results for removals and time."""
    plot_average_metric(
        metric='removals',
        ylabel='Tuples Removed',
        size_results=size_results,
        sizes=sizes,
        epsilon_adjust_factors=epsilon_adjust_factors,
        graphs_folder=graphs_folder,
        current_size_idx=current_size_idx
    )

    plot_average_metric(
        metric='time',
        ylabel='Time Taken (seconds)',
        size_results=size_results,
        sizes=sizes,
        epsilon_adjust_factors=epsilon_adjust_factors,
        graphs_folder=graphs_folder,
        current_size_idx=current_size_idx
    )


# Function to calculate and log averages for all trials
def log_factor_averages(size, size_results, factor_idx, size_stats_file, epsilon_factor):
    """Logs the average results for all algorithms under a specific epsilon factor."""
    header = f"\nAverages for Dataset Size {size}, Epsilon Factor {epsilon_factor}:\n" + "=" * 50 + "\n"
    size_stats_file.write(header)
    log_and_print(header)

    for algo_name in size_results:
        metrics = size_results[algo_name]

        # Retrieve removals and times for the current factor and size
        removals = metrics["removals"][factor_idx][size].get('values', [])
        times = metrics["time"][factor_idx][size].get('values', [])
        successes = metrics["successes"][factor_idx][size].get('values', [])

        # Initialize a dictionary to track the removals by group
        avg_removals_by_group = {}

        # For each group (T, W), calculate the average number of removals
        for group in metrics["removals_by_group"][factor_idx][size]["values"]:
            group_removals = metrics["removals_by_group"][factor_idx][size]["values"][group]
            avg_removals_by_group[group] = np.mean(group_removals) if group_removals else float('nan')

        # Calculate overall metrics
        num_successes = len(successes)
        total_trials = len(removals)
        avg_removals = np.mean(removals) if removals else float('nan')
        avg_time = np.mean(times) if times else float('nan')

        # Write the statistics for the algorithm to the stats file and log it
        stats = (
            f"Algorithm: {algo_name}\n"
            f"Number of Successful Trials: {num_successes}\n"
            f"Average Tuples Removed: {avg_removals:.2f}\n"
            f"Average Time Taken: {avg_time:.2f} seconds\n"
        )

        # Add the average removals for each group to the stats
        for group, avg_removal in avg_removals_by_group.items():
            stats += f"Average Tuples Removed From {group}: {avg_removal:.2f}\n"

        stats += "-" * 50 + "\n"
        size_stats_file.write(stats)
        log_and_print(stats)


def initialize_size_results(epsilon_adjust_factors, database_sizes, confounder_combinations):
    """Initializes the results data structure for storing metrics, with dynamic handling of multiple confounders."""
    algorithms = ["Per Group Removal Algorithm"]
    results = {}
    for algo in algorithms:
        results[algo] = {
            "removals": [{size: {'values': []} for size in database_sizes} for _ in epsilon_adjust_factors],
            "removals_by_group": [{size: {'values': {f'(T={T}, {combo})': [] for T in [0, 1] for combo in confounder_combinations}} for size in database_sizes} for _ in epsilon_adjust_factors],
            "time": [{size: {'values': []} for size in database_sizes} for _ in epsilon_adjust_factors],
            "successes": [{size: {'values': []} for size in database_sizes} for _ in epsilon_adjust_factors]
        }
    return results


def execute_trial(algo_func, dataset_info, target_diff, epsilon, algo_name, trial, factor_idx, size_idx, size, confounder_combinations, confounders):
    """Executes a single trial of an algorithm and returns the results."""
    start_time = time.time()

    # Clone dataset and aggregates to prevent interference
    dataset_copy = dataset_info[0].copy()
    aggregates_copy = {k: v.copy() for k, v in dataset_info[1].items()}

    # Execute the algorithm
    try:
        result = algo_func((dataset_copy, aggregates_copy), target_diff, epsilon, confounder_combinations, confounders)

        if result:
            removals, removals_by_group, iterations, differences = result
        else:
            raise ValueError("Algorithm did not return valid results.")

    except ValueError as e:
        log_and_print(f"Error in algorithm execution: {e}")
        return None

    elapsed_time = time.time() - start_time
    final_diff = differences[-1] if differences else None
    solution_found = final_diff is not None and target_diff - epsilon <= final_diff <= target_diff + epsilon

    return {
        'algo_name': algo_name,
        'removals': removals,
        'removals_by_group': removals_by_group,
        'elapsed_time': elapsed_time,
        'size_idx': size_idx,
        'factor_idx': factor_idx,
        'size': size,
        'target_diff': target_diff,
        'epsilon': epsilon,
        'final_diff': final_diff,
        'solution_found': solution_found,
        'trial': trial
    }


def run_experiment(database_sizes, num_trials=3, target_percentage=None, epsilon_adjust_factors=None, desired_ATE=None, epsilon=None, modify_current_ATE=None, num_confounders=2, confounder_values_per_W=3):
    """Runs the experiment across different dataset sizes and epsilon factors, allowing for either target percentage, desired ATE, or modifying current ATE."""
    if epsilon_adjust_factors is None:
        epsilon_adjust_factors = [10]
    num_cores = cpu_count()
    log_and_print(f"Using {num_cores} CPU cores for parallel processing.")

    # Set random seed
    seed = random.randint(0, 1000000)
    np.random.seed(seed)
    log_and_print(f"\nStarting experiment with random seed {seed}.")

    # Setup directories
    top_level_folder = "experiment_results"
    graphs_folder = os.path.join(top_level_folder, "graphs")
    os.makedirs(graphs_folder, exist_ok=True)

    # Create dataset to determine confounder combinations
    dataset_df = create_dataset(size=100, num_confounders=num_confounders)  # Create a small dataset to get unique confounder combinations
    confounders = [f'W{i+1}' for i in range(num_confounders)]  # Define confounder names
    confounder_combinations = dataset_df.groupby(confounders).size().index

    # Initialize results with confounders
    size_results = initialize_size_results(epsilon_adjust_factors, database_sizes, confounder_combinations)

    # Define algorithms
    algorithms = {
        "Per Group Removal Algorithm": per_group_removal_algorithm
    }

    # Iterate over each dataset size
    for size_idx, size in enumerate(database_sizes):
        log_and_print(f"\nProcessing Dataset Size: {size}\n")

        # Create dataset and folder
        size_folder = os.path.join(top_level_folder, f"experiment_size_{size}")
        os.makedirs(size_folder, exist_ok=True)
        dataset_df = create_dataset(size, num_confounders=num_confounders, confounder_values_per_W=confounder_values_per_W)
        dataset_np = dataset_df.to_records(index=False)

        # Precompute aggregates
        aggregates = calc_initial_aggregates(dataset_df, confounders)
        confounder_combinations = dataset_df.groupby(confounders).size().index
        dataset_info = (dataset_df, aggregates)

        # Open file to log statistics
        with open(os.path.join(size_folder, "size_stats.txt"), "w") as size_stats_file:
            header = f"Statistics for Dataset Size: {size}\n" + "=" * 50 + "\n"
            size_stats_file.write(header)
            log_and_print(header)

            # Iterate over each epsilon adjustment factor
            for factor_idx, epsilon_factor in enumerate(epsilon_adjust_factors):
                log_and_print(f"\nTesting with Epsilon Factor: {epsilon_factor}\n")

                trial_args = []
                for trial in range(1, num_trials + 1):
                    # Option 1: Modify current ATE
                    if modify_current_ATE is not None:
                        current_ATE = calc_ATE_incremental(aggregates, confounder_combinations)
                        target_ATE = current_ATE + modify_current_ATE
                        log_and_print(f"Current ATE: {current_ATE}, Modified ATE: {target_ATE}, Epsilon: {epsilon}")
                        epsilon_calculated = epsilon

                    # Option 2: Use the desired ATE range
                    elif desired_ATE is not None and epsilon is not None:
                        current_ATE = calc_ATE_incremental(aggregates, confounder_combinations)
                        log_and_print(f"Current ATE: {current_ATE}, Desired ATE: {desired_ATE}, Epsilon: {epsilon}")

                        # The desired range is [desired_ATE - epsilon, desired_ATE + epsilon]
                        target_ATE = desired_ATE
                        epsilon_calculated = epsilon

                    # Option 3: Use target percentage method
                    else:
                        # Randomly select one of the confounder combinations instead of hardcoding to one
                        chosen_confounder_combo = confounder_combinations[np.random.randint(0, len(confounder_combinations))]
                        log_and_print(f"Chosen confounder combination for removal: {chosen_confounder_combo}")

                        # Set a random seed and sample rows only from the chosen confounder combination
                        trial_seed = random.randint(0, 1000000)
                        np.random.seed(trial_seed)

                        confounder_mask_indices = dataset_df[(dataset_df[confounders] == chosen_confounder_combo).all(axis=1)].sample(
                            frac=target_percentage, random_state=trial_seed
                        ).index

                        masked_dataset_df = dataset_df.drop(confounder_mask_indices)
                        masked_aggregates = calc_initial_aggregates(masked_dataset_df, confounders)

                        # Calculate the target ATE from the masked dataset
                        target_ATE = calc_ATE_incremental(masked_aggregates, confounder_combinations)

                        # Calculate dynamic epsilon based on the difference between current ATE and target ATE
                        epsilon_calculated = abs(
                            target_ATE - calc_ATE_incremental(aggregates, confounder_combinations)) / epsilon_factor

                    # Prepare arguments for parallel processing, ensuring 'confounders' is passed
                    for algo_name, algo_func in algorithms.items():
                        trial_args.append(
                            (algo_func, dataset_info, target_ATE, epsilon_calculated,
                             algo_name, trial, factor_idx, size_idx, size, confounder_combinations, confounders)
                        )

                # Execute all trials in parallel
                with Pool(processes=num_cores) as pool:
                    try:
                        results = pool.starmap(execute_trial, trial_args)
                    except Exception as e:
                        log_and_print(f"An error occurred during multiprocessing: {e}", level='error')
                        continue

                # Store the results for both algorithms
                for res in results:
                    algo = res['algo_name']
                    if res['solution_found']:
                        size_results[algo]["removals"][res['factor_idx']][res['size']]["values"].append(res['removals'])
                        for group, removal_count in res['removals_by_group'].items():
                            size_results[algo]["removals_by_group"][res['factor_idx']][res['size']]["values"][group].append(removal_count)
                        size_results[algo]["time"][res['factor_idx']][res['size']]["values"].append(res['elapsed_time'])
                        size_results[algo]["successes"][res['factor_idx']][res['size']]["values"].append(True)
                    else:
                        log_and_print(f"Algorithm {algo} failed to find a solution in Trial {res['trial']}.")

                # Log the average results for the current epsilon factor
                log_factor_averages(size, size_results, factor_idx, size_stats_file, epsilon_factor)

        # Generate plots after all dataset sizes are processed
        plot_experiment_results(size_results, database_sizes, epsilon_adjust_factors, graphs_folder, size_idx)


# Main function to run the experiment
def main():
    # Experiment configuration parameters
    database_sizes = [100000, 200000, 300000]
    epsilon_adjust_factors = [10, 100, 1000, 10000]
    num_trials = 3
    num_confounders = 2
    confounders_values_per_W = 3


    # Option 1: Remove a part of the data randomly in order to get the desired ATE
    target_percentage = 0.1  

    # Option 2: Define desired ATE and epsilon
    desired_ATE = None

    # Option 3: Define a modification to the current ATE
    modify_current_ATE = None

    epsilon = 0.1

    # Log and print the experiment configuration
    config_message = (
        "\nExperiment Configuration:\n"
        f"Database Sizes: {database_sizes}\n"
        f"Number of Trials: {num_trials}\n"
        f"Epsilon Adjustment Factors: {epsilon_adjust_factors}\n"
        f"Target Percentage of Rows to Remove: {target_percentage * 100}%\n"
        f"Desired ATE: {desired_ATE}, Epsilon: {epsilon}\n"
        f"Modify Current ATE By: {modify_current_ATE}\n"
        f"Number of Confounders: {num_confounders}\n"
    )
    log_and_print(config_message)

    # Start the experiment with the chosen approach
    run_experiment(database_sizes, num_trials, target_percentage, epsilon_adjust_factors, desired_ATE, epsilon, modify_current_ATE, num_confounders, confounders_values_per_W)

    log_and_print("\nExperiment Completed.")


if __name__ == "__main__":
    main()
