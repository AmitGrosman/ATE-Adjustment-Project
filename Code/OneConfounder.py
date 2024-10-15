import os
import time
import random
import logging
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
def create_dataset(size=50, loc_1=70, loc_0=40):
    """Generates a dataset with binary treatment and confounder W."""
    T = np.random.randint(0, 2, size)
    O = np.where(
        T == 1,
        np.random.normal(loc=loc_1, scale=20, size=size),
        np.random.normal(loc=loc_0, scale=20, size=size)
    )
    O = np.clip(O, 0, 100)
    W = np.random.randint(0, 2, size)
    return pd.DataFrame({'T': T, 'O': O, 'W': W})

# ATE Calculation Functions

def calc_initial_aggregates(dataset):
    """Calculates sums and counts for each subgroup."""
    aggregates = {}
    for T in [0, 1]:
        for W in [0, 1]:
            subgroup = dataset[(dataset['T'] == T) & (dataset['W'] == W)]
            aggregates[(T, W)] = {'sum_O': subgroup['O'].sum(), 'count': len(subgroup)}
    return aggregates

def calc_ATE_incremental(aggregates):
    """Calculates the ATE based on aggregates."""
    E_Y_T1_W1 = aggregates[(1, 1)]['sum_O'] / aggregates[(1, 1)]['count'] if aggregates[(1, 1)]['count'] > 0 else 0
    E_Y_T1_W0 = aggregates[(1, 0)]['sum_O'] / aggregates[(1, 0)]['count'] if aggregates[(1, 0)]['count'] > 0 else 0
    E_Y_T0_W1 = aggregates[(0, 1)]['sum_O'] / aggregates[(0, 1)]['count'] if aggregates[(0, 1)]['count'] > 0 else 0
    E_Y_T0_W0 = aggregates[(0, 0)]['sum_O'] / aggregates[(0, 0)]['count'] if aggregates[(0, 0)]['count'] > 0 else 0

    total_count = sum([aggregates[key]['count'] for key in aggregates])
    p_w1 = (aggregates[(1, 1)]['count'] + aggregates[(0, 1)]['count']) / total_count if total_count > 0 else 0
    p_w0 = 1 - p_w1

    return p_w1 * (E_Y_T1_W1 - E_Y_T0_W1) + p_w0 * (E_Y_T1_W0 - E_Y_T0_W0)

def calc_ATE_with_mask(aggregates, dataset, mask):
    """Recalculates ATE with the mask applied."""
    masked_indices_T1_W1 = (dataset['T'] == 1) & (dataset['W'] == 1) & mask
    masked_indices_T1_W0 = (dataset['T'] == 1) & (dataset['W'] == 0) & mask
    masked_indices_T0_W1 = (dataset['T'] == 0) & (dataset['W'] == 1) & mask
    masked_indices_T0_W0 = (dataset['T'] == 0) & (dataset['W'] == 0) & mask

    E_Y_T1_W1 = dataset['O'][masked_indices_T1_W1].mean() if np.sum(masked_indices_T1_W1) > 0 else 0
    E_Y_T1_W0 = dataset['O'][masked_indices_T1_W0].mean() if np.sum(masked_indices_T1_W0) > 0 else 0
    E_Y_T0_W1 = dataset['O'][masked_indices_T0_W1].mean() if np.sum(masked_indices_T0_W1) > 0 else 0
    E_Y_T0_W0 = dataset['O'][masked_indices_T0_W0].mean() if np.sum(masked_indices_T0_W0) > 0 else 0

    total_count = np.sum(mask)
    p_w1 = np.sum(masked_indices_T1_W1 | masked_indices_T0_W1) / total_count if total_count > 0 else 0
    p_w0 = 1 - p_w1

    return p_w1 * (E_Y_T1_W1 - E_Y_T0_W1) + p_w0 * (E_Y_T1_W0 - E_Y_T0_W0)

def ATE_after_row_removal(row_idx, dataset, mask, aggregates):
    """Calculates ATE as if the row was removed."""
    temp_mask = mask.copy()
    temp_mask[row_idx] = False
    return calc_ATE_with_mask(aggregates, dataset, temp_mask)


def calculate_row_impact_and_sort(dataset, mask, aggregates, target_ATE):
    """Helper to calculate and sort rows by their impact on ATE."""
    row_diffs = []
    for i in range(len(dataset)):
        if mask[i]:
            new_ATE = ATE_after_row_removal(i, dataset, mask, aggregates)
            diff = abs(new_ATE - target_ATE)
            row_diffs.append((i, diff))
    return sorted(row_diffs, key=lambda x: x[1])


# Algorithm 1: Per Group Removal Algorithm
def per_group_removal_algorithm(dataset_info, target_ATE, epsilon):
    """Algorithm to remove rows per group (W and T) and select the one affecting ATE the most."""
    dataset, aggregates = dataset_info
    mask = np.ones(len(dataset), dtype=bool)
    total_removals, removals_W0, removals_W1 = 0, 0, 0
    iterations, differences = [], []

    # Sort dataset by O for each of the four groups
    sorted_T0_W0_indices = np.where((dataset['T'] == 0) & (dataset['W'] == 0))[0][np.argsort(dataset['O'][(dataset['T'] == 0) & (dataset['W'] == 0)])]
    sorted_T0_W1_indices = np.where((dataset['T'] == 0) & (dataset['W'] == 1))[0][np.argsort(dataset['O'][(dataset['T'] == 0) & (dataset['W'] == 1)])]
    sorted_T1_W0_indices = np.where((dataset['T'] == 1) & (dataset['W'] == 0))[0][np.argsort(dataset['O'][(dataset['T'] == 1) & (dataset['W'] == 0)])]
    sorted_T1_W1_indices = np.where((dataset['T'] == 1) & (dataset['W'] == 1))[0][np.argsort(dataset['O'][(dataset['T'] == 1) & (dataset['W'] == 1)])]

    current_ATE = calc_ATE_with_mask(aggregates, dataset, mask)

    while True:
        best_diff, row_to_remove = float('inf'), None

        # Check ATE and decide whether to remove highest or lowest from each group
        if current_ATE > target_ATE + epsilon:
            # ATE is above target, remove highest from each group
            candidates = []
            if np.any(mask[sorted_T0_W0_indices]):
                candidates.append(sorted_T0_W0_indices[mask[sorted_T0_W0_indices]][-1])  # Highest T=0, W=0
            if np.any(mask[sorted_T0_W1_indices]):
                candidates.append(sorted_T0_W1_indices[mask[sorted_T0_W1_indices]][-1])  # Highest T=0, W=1
            if np.any(mask[sorted_T1_W0_indices]):
                candidates.append(sorted_T1_W0_indices[mask[sorted_T1_W0_indices]][-1])  # Highest T=1, W=0
            if np.any(mask[sorted_T1_W1_indices]):
                candidates.append(sorted_T1_W1_indices[mask[sorted_T1_W1_indices]][-1])  # Highest T=1, W=1
        else:
            # ATE is below target, remove lowest from each group
            candidates = []
            if np.any(mask[sorted_T0_W0_indices]):
                candidates.append(sorted_T0_W0_indices[mask[sorted_T0_W0_indices]][0])  # Lowest T=0, W=0
            if np.any(mask[sorted_T0_W1_indices]):
                candidates.append(sorted_T0_W1_indices[mask[sorted_T0_W1_indices]][0])  # Lowest T=0, W=1
            if np.any(mask[sorted_T1_W0_indices]):
                candidates.append(sorted_T1_W0_indices[mask[sorted_T1_W0_indices]][0])  # Lowest T=1, W=0
            if np.any(mask[sorted_T1_W1_indices]):
                candidates.append(sorted_T1_W1_indices[mask[sorted_T1_W1_indices]][0])  # Lowest T=1, W=1

        # Simulate removing the selected rows from each group and find the best one
        for candidate in candidates:
            new_ATE = ATE_after_row_removal(candidate, dataset, mask, aggregates)
            diff = abs(new_ATE - target_ATE)
            if diff < best_diff:
                best_diff = diff
                row_to_remove = candidate

        # Stop if no rows to remove
        if row_to_remove is None:
            break

        # Update mask to remove the chosen row
        mask[row_to_remove] = False
        total_removals += 1
        if dataset['W'][row_to_remove] == 0:
            removals_W0 += 1
        else:
            removals_W1 += 1

        # Update current ATE
        current_ATE = calc_ATE_with_mask(aggregates, dataset, mask)
        iterations.append(total_removals)
        differences.append(current_ATE)

        # Check if the target ATE has been reached within the epsilon range
        if target_ATE - epsilon <= current_ATE <= target_ATE + epsilon:
            break

    return total_removals, removals_W0, removals_W1, iterations, differences


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
        removals_W0 = metrics["removals_W0"][factor_idx][size].get('values', [])
        removals_W1 = metrics["removals_W1"][factor_idx][size].get('values', [])
        times = metrics["time"][factor_idx][size].get('values', [])
        successes = metrics["successes"][factor_idx][size].get('values', [])

        num_successes = len(successes)
        total_trials = len(removals)
        if total_trials > 0:
            avg_removals = np.mean(removals) if removals else float('nan')
            avg_removals_W0 = np.mean(removals_W0) if removals_W0 else float('nan')
            avg_removals_W1 = np.mean(removals_W1) if removals_W1 else float('nan')
            avg_time = np.mean(times) if times else float('nan')
        else:
            avg_removals = float('nan')
            avg_removals_W0 = float('nan')
            avg_removals_W1 = float('nan')
            avg_time = float('nan')

        stats = (
            f"Algorithm: {algo_name}\n"
            f"Number of Successful Trials: {num_successes}\n"
            f"Average Tuples Removed: {avg_removals:.2f}\n"
            f"Average Tuples Removed From W=0: {avg_removals_W0:.2f}\n"
            f"Average Tuples Removed From W=1: {avg_removals_W1:.2f}\n"
            f"Average Time Taken: {avg_time:.2f} seconds\n"
            + "-" * 50 + "\n"
        )
        size_stats_file.write(stats)
        log_and_print(stats)


# Results Initialization
def initialize_size_results(epsilon_adjust_factors, database_sizes):
    """Initializes the results data structure for storing metrics."""
    algorithms = ["Per Group Removal Algorithm"]
    results = {}
    for algo in algorithms:
        results[algo] = {
            "removals": [{size: {'values': []} for size in database_sizes} for _ in epsilon_adjust_factors],
            "removals_W0": [{size: {'values': []} for size in database_sizes} for _ in epsilon_adjust_factors],
            "removals_W1": [{size: {'values': []} for size in database_sizes} for _ in epsilon_adjust_factors],
            "time": [{size: {'values': []} for size in database_sizes} for _ in epsilon_adjust_factors],
            "successes": [{size: {'values': []} for size in database_sizes} for _ in epsilon_adjust_factors]
        }
    return results


# Trial Execution
def execute_trial(algo_func, dataset_info, target_diff, epsilon, algo_name, trial, factor_idx, size_idx, size):
    """Executes a single trial of an algorithm and returns the results."""
    start_time = time.time()

    # Clone dataset and aggregates to prevent interference
    dataset_copy = dataset_info[0].copy()
    aggregates_copy = {k: v.copy() for k, v in dataset_info[1].items()}

    # Execute the algorithm
    result = algo_func((dataset_copy, aggregates_copy), target_diff, epsilon)

    elapsed_time = time.time() - start_time

    removals, removals_W0, removals_W1, iterations, differences = result
    final_diff = differences[-1] if differences else None
    solution_found = final_diff is not None and target_diff - epsilon <= final_diff <= target_diff + epsilon

    return {
        'algo_name': algo_name,
        'removals': removals,
        'removals_W0': removals_W0,
        'removals_W1': removals_W1,
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


def run_experiment(database_sizes, num_trials=3, target_percentage=None,
                   epsilon_adjust_factors=None):
    """Runs the experiment across different dataset sizes and epsilon factors with dynamic epsilon."""
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

    # Initialize results
    size_results = initialize_size_results(epsilon_adjust_factors, database_sizes)

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
        dataset_df = create_dataset(size)
        dataset_np = dataset_df.to_records(index=False)
        dataset_np = np.array(dataset_np)

        # Precompute aggregates
        aggregates = calc_initial_aggregates(dataset_df)
        dataset_info = (dataset_np, aggregates)

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
                    # Calculate the target ATE based on the mask
                    trial_seed = random.randint(0, 1000000)
                    np.random.seed(trial_seed)
                    W0_mask_indices = dataset_df[dataset_df['W'] == 0].sample(
                        frac=target_percentage, random_state=trial_seed
                    ).index
                    masked_dataset_df = dataset_df.drop(W0_mask_indices)
                    masked_aggregates = calc_initial_aggregates(masked_dataset_df)

                    # Calculate the target ATE from the masked dataset
                    target_ATE = calc_ATE_incremental(masked_aggregates)

                    # Calculate dynamic epsilon based on the difference between current ATE and target ATE
                    epsilon_calculated = abs(target_ATE - calc_ATE_incremental(aggregates)) / epsilon_factor

                    # Prepare arguments for parallel processing
                    for algo_name, algo_func in algorithms.items():
                        trial_args.append(
                            (algo_func, dataset_info, target_ATE, epsilon_calculated,
                             algo_name, trial, factor_idx, size_idx, size)
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
                        size_results[algo]["removals_W0"][res['factor_idx']][res['size']]["values"].append(res['removals_W0'])
                        size_results[algo]["removals_W1"][res['factor_idx']][res['size']]["values"].append(res['removals_W1'])
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
    database_sizes = [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000]
    epsilon_adjust_factors = [10, 100, 1000, 10000]
    num_trials = 3
    target_percentage = 0.1

    # Log and print the experiment configuration
    config_message = (
        "\nExperiment Configuration:\n"
        f"Database Sizes: {database_sizes}\n"
        f"Number of Trials: {num_trials}\n"
        f"Epsilon Adjustment Factors: {epsilon_adjust_factors}\n"
        f"Target Percentage of Rows to Remove: {target_percentage * 100}%\n"
    )
    log_and_print(config_message)

    # Start the experiment
    run_experiment(database_sizes, num_trials, target_percentage, epsilon_adjust_factors)

    log_and_print("\nExperiment Completed.")

if __name__ == "__main__":
    main()
