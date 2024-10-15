import os
import time
import random
import logging
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging to record experiment details and debug information
logging.basicConfig(
    filename='experiment.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def log_and_print(message, level='info'):
    """
    Helper function to log messages to a file and print them to the console simultaneously.
    """
    getattr(logging, level)(message)  # Log the message with the specified level
    print(message)  # Print the message to the console


def create_dataset(size=50, loc_1=70, loc_0=40):
    """
    Generates a synthetic dataset with binary treatment assignment and outcome values.
    """
    T = np.random.randint(0, 2, size)  # Randomly assign group 1 or group 0
    O = np.where(T == 1,
                 np.random.normal(loc_1, scale=20, size=size),  # Outcomes for T=1
                 np.random.normal(loc_0, scale=20, size=size))  # Outcomes for T=0
    O = np.clip(O, 0, 100)  # Ensure all outcome values are within [0, 100]
    return pd.DataFrame({'T': T, 'O': O})


def calc_avg(dataset):
    """
    Calculates the absolute difference between the average outcomes of treatment and control groups.
    """
    avg_T1 = dataset.loc[dataset['T'] == 1, 'O'].mean()
    avg_T0 = dataset.loc[dataset['T'] == 0, 'O'].mean()
    if pd.isna(avg_T1) or pd.isna(avg_T0):
        return None
    return abs(avg_T1 - avg_T0)


def execute_trial(algo_func, dataset, target_diff, epsilon, algo_name, trial, factor_idx, size_idx, size):
    """
    Executes a single trial of an algorithm and returns the results.
    Note: Per-trial logging is removed to comply with user requirements.
    """

    start_time = time.time()
    result, removals, iterations, differences = algo_func(dataset.copy(), target_diff, epsilon)
    elapsed_time = time.time() - start_time
    final_diff = calc_avg(result)


    return {
        'algo_name': algo_name,
        'removals': removals,
        'elapsed_time': elapsed_time,
        'size_idx': size_idx,
        'factor_idx': factor_idx,
        'size': size,
        'target_diff': target_diff,
        'epsilon': epsilon,
        'final_diff': final_diff
    }


def initialize_groups(dataset):
    """
    Initializes and sorts treatment and control groups, calculating initial sums and counts.
    """
    D0 = dataset[dataset['T'] == 0].sort_values(by='O').reset_index(drop=True)
    D1 = dataset[dataset['T'] == 1].sort_values(by='O').reset_index(drop=True)
    sum_T0, sum_T1 = D0['O'].sum(), D1['O'].sum()
    n_T0, n_T1 = len(D0), len(D1)
    if n_T0 > 0 and n_T1 > 0:
        current_diff = abs((sum_T1 / n_T1) - (sum_T0 / n_T0))
    else:
        current_diff = None  # Stop if one group is empty
    return D0, D1, sum_T0, sum_T1, n_T0, n_T1, current_diff


def greedy_naive_algorithm(dataset, target_diff, epsilon):
    """
    Greedy Naive algorithm that iteratively removes the highest 'O' from the group that exceeds the target difference.
    """
    D0, D1, sum_T0, sum_T1, n_T0, n_T1, current_diff = initialize_groups(dataset)
    count = 0
    iterations, differences = [], []

    # Loop until the difference is within the acceptable range
    while (n_T0 > 0 and n_T1 > 0) and (current_diff is not None and
                                       (current_diff < target_diff - epsilon or current_diff > target_diff + epsilon)):
        if current_diff > target_diff + epsilon:
            if n_T1 > 0:
                removed_O = D1.iloc[-1]['O']  # Remove the highest 'O' from D1
                sum_T1 -= removed_O
                D1 = D1.iloc[:-1]
                n_T1 -= 1
        else:
            if n_T0 > 0:
                removed_O = D0.iloc[-1]['O']  # Remove the highest 'O' from D0
                sum_T0 -= removed_O
                D0 = D0.iloc[:-1]
                n_T0 -= 1

        # Recalculate the difference after removal
        if n_T0 > 0 and n_T1 > 0:
            current_diff = abs((sum_T1 / n_T1) - (sum_T0 / n_T0))
        else:
            current_diff = None  # Stop if one group is empty

        count += 1
        iterations.append(count)
        differences.append(current_diff)

    if current_diff is None:
        return dataset, len(dataset), iterations, differences  # All rows removed

    adjusted_dataset = pd.concat([D0, D1]).reset_index(drop=True)
    return adjusted_dataset, count, iterations, differences


def greedy_binary_search_algorithm(dataset, target_diff, epsilon):
    """
    Greedy Binary Search algorithm that uses binary search to find the optimal row to remove.
    """
    D0, D1, sum_T0, sum_T1, n_T0, n_T1, current_diff = initialize_groups(dataset)
    count = 0
    iterations, differences = [], []

    # Loop until the difference is within the acceptable range
    while (n_T0 > 0 and n_T1 > 0) and (current_diff is not None and
                                       (current_diff < target_diff - epsilon or current_diff > target_diff + epsilon)):
        best_diff = float('inf')
        best_index = None
        is_D1 = False

        if current_diff > target_diff + epsilon:
            # Binary search in D1 to find the best 'O' to remove
            left, right = 0, n_T1 - 1
            while left <= right:
                mid = (left + right) // 2
                temp_sum_T1 = sum_T1 - D1.loc[mid, 'O']
                temp_n_T1 = n_T1 - 1
                temp_diff = abs((temp_sum_T1 / temp_n_T1) - (sum_T0 / n_T0)) if temp_n_T1 > 0 else None

                if temp_diff is not None and abs(temp_diff - target_diff) < abs(best_diff - target_diff):
                    best_diff = temp_diff
                    best_index = mid
                    is_D1 = True

                if temp_diff is not None and temp_diff > target_diff:
                    left = mid + 1
                else:
                    right = mid - 1
        else:
            # Binary search in D0 to find the best 'O' to remove
            left, right = 0, n_T0 - 1
            while left <= right:
                mid = (left + right) // 2
                temp_sum_T0 = sum_T0 - D0.loc[mid, 'O']
                temp_n_T0 = n_T0 - 1
                temp_diff = abs((sum_T1 / n_T1) - (temp_sum_T0 / temp_n_T0)) if temp_n_T0 > 0 else None

                if temp_diff is not None and abs(temp_diff - target_diff) < abs(best_diff - target_diff):
                    best_diff = temp_diff
                    best_index = mid
                    is_D1 = False

                if temp_diff is not None and temp_diff < target_diff:
                    left = mid + 1
                else:
                    right = mid - 1

        if best_index is None:
            break  # No optimal row found

        # Remove the identified row from the appropriate group
        if is_D1:
            removed_O = D1.loc[best_index, 'O']
            sum_T1 -= removed_O
            D1 = D1.drop(best_index).reset_index(drop=True)
            n_T1 -= 1
        else:
            removed_O = D0.loc[best_index, 'O']
            sum_T0 -= removed_O
            D0 = D0.drop(best_index).reset_index(drop=True)
            n_T0 -= 1

        # Recalculate the difference after removal
        if n_T0 > 0 and n_T1 > 0:
            current_diff = abs((sum_T1 / n_T1) - (sum_T0 / n_T0))
        else:
            current_diff = None  # Stop if one group is empty

        count += 1
        iterations.append(count)
        differences.append(current_diff)

    if current_diff is None:
        return dataset, len(dataset), iterations, differences  # All rows removed

    adjusted_dataset = pd.concat([D0, D1]).reset_index(drop=True)
    return adjusted_dataset, count, iterations, differences


def combined_naive_binary_search_algorithm(dataset, target_diff, epsilon):
    """
    Combines the Greedy Naive and Greedy Binary Search algorithms for optimization.
    """
    D0, D1, sum_T0, sum_T1, n_T0, n_T1, current_diff = initialize_groups(dataset)
    count = 0
    iterations, differences = [], []
    switched_to_binary = False
    previous_diff_minus_target = current_diff - target_diff if current_diff is not None else None

    # Naive approach loop
    while (n_T0 > 0 and n_T1 > 0) and (current_diff is not None and
                                       (current_diff < target_diff - epsilon or current_diff > target_diff + epsilon)):
        if current_diff > target_diff + epsilon:
            if n_T1 > 0:
                removed_O = D1.iloc[-1]['O']  # Remove the highest 'O' from D1
                sum_T1 -= removed_O
                D1 = D1.iloc[:-1]
                n_T1 -= 1
        else:
            if n_T0 > 0:
                removed_O = D0.iloc[-1]['O']  # Remove the highest 'O' from D0
                sum_T0 -= removed_O
                D0 = D0.iloc[:-1]
                n_T0 -= 1

        # Recalculate the difference after removal
        if n_T0 > 0 and n_T1 > 0:
            current_diff = abs((sum_T1 / n_T1) - (sum_T0 / n_T0))
        else:
            current_diff = None  # Stop if one group is empty

        count += 1
        iterations.append(count)
        differences.append(current_diff)

        if current_diff is None:
            break

        # Check if the sign of (current_diff - target_diff) has changed to switch to binary search
        current_diff_minus_target = current_diff - target_diff
        if previous_diff_minus_target * current_diff_minus_target < 0:
            switched_to_binary = True
            break  # Exit the loop to switch to binary search

        previous_diff_minus_target = current_diff_minus_target

    if current_diff is None:
        return dataset, len(dataset), iterations, differences  # All rows removed

    if switched_to_binary:
        # Switch to binary search for fine-tuning
        combined_dataset = pd.concat([D0, D1]).reset_index(drop=True)
        result, bs_count, bs_iterations, bs_differences = greedy_binary_search_algorithm(
            combined_dataset, target_diff, epsilon)
        total_count = count + bs_count
        total_iterations = iterations + bs_iterations
        total_differences = differences + bs_differences
        return result, total_count, total_iterations, total_differences
    else:
        # Combine the adjusted groups
        adjusted_dataset = pd.concat([D0, D1]).reset_index(drop=True)
        return adjusted_dataset, count, iterations, differences


def log_trial_summary(trial_num, size, original_diff, new_diff, epsilon, epsilon_factor, size_stats_file):
    """
    Logs the summary of a single trial to both the file and console.
    """
    trial_header = f"\n--- Trial {trial_num} ---\n"
    summary = (
        f"Original Difference: {original_diff:.6f}\n"
        f"New Difference: {new_diff:.6f}\n"
        f"Epsilon Factor: {epsilon_factor}\n"
        f"Epsilon: {epsilon:.10f}\n\n"
    )
    size_stats_file.write(trial_header + summary)
    log_and_print(trial_header + summary)


def log_factor_averages(size, size_results, factor_idx, size_stats_file, epsilon_factor):
    """
    Logs the average results for each algorithm under a specific epsilon factor.
    """
    header = f"\nAverages for Dataset Size {size}, Epsilon Factor {epsilon_factor}:\n" + "=" * 50 + "\n"
    size_stats_file.write(header)
    log_and_print(header)

    for algo_name, metrics in size_results.items():
        # Retrieve removals and times for the current factor and size
        removals = metrics["removals"][factor_idx][size].get('values', [])
        times = metrics["time"][factor_idx][size].get('values', [])

        # Calculate averages, handling cases with no valid data
        avg_removals = np.mean(removals) if removals else size
        avg_time = np.mean(times) if times else 0

        # Prepare the statistics string
        stats = (
                f"Algorithm: {algo_name}\n"
                f"Average Tuples Removed: {avg_removals}\n"
                f"Average Time Taken: {avg_time:.2f} seconds\n"
                + "-" * 50 + "\n"
        )
        size_stats_file.write(stats)
        log_and_print(stats)


def plot_average_metric(metric, ylabel, size_results, sizes, epsilon_adjust_factors, graphs_folder,
                        current_size_idx):
    """
    Plots the average metric (either removals or time) against dataset sizes for each algorithm and epsilon factor.
    Only includes data up to the current_size_idx.
    """
    sns.set(style="whitegrid")  # Seaborn style
    for factor_idx, epsilon_factor in enumerate(epsilon_adjust_factors):
        plt.figure(figsize=(12, 8))
        for algo_name, metrics in size_results.items():
            # Compute average values for each size up to current_size_idx
            avg_values = []
            for i in range(current_size_idx + 1):
                size = sizes[i]
                values = metrics[metric][factor_idx][size].get('values', [])
                if values:
                    avg = np.mean(values)
                else:
                    avg = size  # Assign total size if all rows are removed
                avg_values.append(avg)
            plt.plot(sizes[:current_size_idx + 1], avg_values, marker='o', label=algo_name)

        # Configure plot aesthetics and labels
        plt.title(f'Average {ylabel} vs Dataset Size (Epsilon Factor: {epsilon_factor})', fontsize=16)
        plt.xlabel('Dataset Size', fontsize=14)
        plt.ylabel(f'Average {ylabel}', fontsize=14)
        plt.legend(title='Algorithms')
        plt.xticks(sizes[:current_size_idx + 1])
        plt.grid(True)
        plt.tight_layout()

        # Create directory for the specific metric and save the plot
        metric_folder = os.path.join(graphs_folder, f"{metric}")
        os.makedirs(metric_folder, exist_ok=True)
        plt.savefig(os.path.join(metric_folder, f"average_{metric}_vs_size_eps_{epsilon_factor}.png"))
        plt.close()


def plot_experiment_results(size_results, sizes, epsilon_adjust_factors, graphs_folder, current_size_idx):
    """
    Plots all experiment results, including average removals and time taken for each algorithm and epsilon factor.
    """
    # Plot average tuples removed
    plot_average_metric(
        metric='removals',
        ylabel='Tuples Removed',
        size_results=size_results,
        sizes=sizes,
        epsilon_adjust_factors=epsilon_adjust_factors,
        graphs_folder=graphs_folder,
        current_size_idx=current_size_idx
    )

    # Plot average time taken
    plot_average_metric(
        metric='time',
        ylabel='Time Taken (seconds)',
        size_results=size_results,
        sizes=sizes,
        epsilon_adjust_factors=epsilon_adjust_factors,
        graphs_folder=graphs_folder,
        current_size_idx=current_size_idx
    )


def initialize_size_results(epsilon_adjust_factors, database_sizes):
    """
    Initializes the results data structure for storing removals and time metrics for each algorithm.
    """
    return {
        algo: {
            "removals": [{size: {'values': []} for size in database_sizes} for _ in epsilon_adjust_factors],
            "time": [{size: {'values': []} for size in database_sizes} for _ in epsilon_adjust_factors]
        }
        for algo in ["Greedy Naive", "Greedy Binary Search", "Combined Naive-Binary"]
    }


def run_experiment(database_sizes, num_trials=3, epsilon=0.01, target_diff=None, target_percentage=None,
                   epsilon_adjust_factors=None, use_dynamic_epsilon=True):
    """
    Runs the experiment across different dataset sizes, trials, and epsilon adjustment factors.
    Utilizes multiprocessing for parallel trial executions.
    """
    if epsilon_adjust_factors is None:
        epsilon_adjust_factors = [4]
    num_cores = cpu_count()
    log_and_print(f"Using {num_cores} CPU cores for parallel processing.")

    # Setup directories for experiment results and temporary graphs
    top_level_folder = "experiment_results"
    graphs_folder = os.path.join(top_level_folder, "graphs")
    os.makedirs(graphs_folder, exist_ok=True)

    # Initialize the results data structure
    size_results = initialize_size_results(epsilon_adjust_factors, database_sizes)

    # Define the algorithms to be tested
    algorithms = {
        "Greedy Naive": greedy_naive_algorithm,
        "Greedy Binary Search": greedy_binary_search_algorithm,
        "Combined Naive-Binary": combined_naive_binary_search_algorithm
    }

    # Iterate over each dataset size
    for size_idx, size in enumerate(database_sizes):
        # Set a random seed for reproducibility
        seed = random.randint(0, 1000000)
        np.random.seed(seed)
        log_and_print(f"\nStarting experiment with random seed {seed} for dataset size {size}.")

        # Create dataset and corresponding folder
        size_folder = os.path.join(top_level_folder, f"experiment_size_{size}")
        os.makedirs(size_folder, exist_ok=True)
        dataset = create_dataset(size)
        dataset.to_csv(os.path.join(size_folder, "full_dataset.csv"), index=False)

        # Open a file to log statistics for the current dataset size
        with open(os.path.join(size_folder, "size_stats.txt"), "w") as size_stats_file:
            header = f"Statistics for Dataset Size: {size}\n" + "=" * 50 + "\n"
            size_stats_file.write(header)
            log_and_print(header)

            # Iterate over each epsilon adjustment factor
            for factor_idx, epsilon_factor in enumerate(epsilon_adjust_factors):
                log_and_print(f"\nTesting with Epsilon Factor: {epsilon_factor}\n")

                trial_args = []
                for trial in range(1, num_trials + 1):
                    # Calculate the original difference before any removals
                    original_diff = calc_avg(dataset)

                    # Adjust target difference based on target percentage if applicable
                    if target_percentage is not None and use_dynamic_epsilon:
                        num_removals = int(target_percentage * size)
                        removed_indices = np.random.choice(dataset.index, size=num_removals, replace=False)
                        remaining_data = dataset.drop(removed_indices)
                        new_diff = calc_avg(remaining_data)
                    else:
                        new_diff = target_diff

                    # Calculate epsilon based on the difference adjustment
                    if new_diff is not None:
                        epsilon_calculated = abs(original_diff - new_diff) / epsilon_factor
                    else:
                        epsilon_calculated = epsilon

                    # Log the trial summary (only the averages will be logged later)
                    log_trial_summary(
                        trial_num=trial,
                        size=size,
                        original_diff=original_diff if original_diff is not None else 0,
                        new_diff=new_diff if new_diff is not None else 0,
                        epsilon=epsilon_calculated,
                        epsilon_factor=epsilon_factor,
                        size_stats_file=size_stats_file
                    )

                    # Prepare arguments for parallel processing of each algorithm
                    for algo_name, algo_func in algorithms.items():
                        trial_args.append(
                            (algo_func, dataset.copy(), new_diff, epsilon_calculated, algo_name,
                             trial, factor_idx, size_idx, size)
                        )

                # Execute all trials in parallel using multiprocessing.starmap
                with Pool(processes=num_cores) as pool:
                    results = pool.starmap(execute_trial, trial_args)

                # Store the results from each trial
                for res in results:
                    algo = res['algo_name']
                    size_results[algo]["removals"][factor_idx][res['size']]["values"].append(res['removals'])
                    size_results[algo]["time"][factor_idx][res['size']]["values"].append(res['elapsed_time'])

                # Log the average results for the current epsilon factor
                log_factor_averages(size, size_results, factor_idx, size_stats_file, epsilon_factor)

        # After processing the current dataset size, generate plots
        plot_experiment_results(
            size_results=size_results,
            sizes=database_sizes,
            epsilon_adjust_factors=epsilon_adjust_factors,
            graphs_folder=graphs_folder,
            current_size_idx=size_idx
        )


def main():
    """
    Main function to configure and start the experiment.
    """
    # Experiment configuration parameters
    database_sizes = [1000000, 2000000, 3000000, 4000000, 5000000,
                      6000000, 7000000, 8000000, 9000000, 10000000]
    num_trials = 3

    epsilon_adjust_factors = [10, 100, 1000, 10000, 100000]
    target_percentage = 0.1

    target_diff = None  # If None, target_diff will be determined based on target_percentage
    epsilon = 0.01

    use_dynamic_epsilon = True

    # Log and print the experiment configuration
    config_message = (
        "\nExperiment Configuration:\n"
        f"Database Sizes: {database_sizes}\n"
        f"Number of Trials: {num_trials}\n"
        f"Use Dynamic Epsilon: {use_dynamic_epsilon}\n"
        f"Epsilon Adjustment Factors: {epsilon_adjust_factors}\n"
        f"Target Percentage of Rows to Remove: {target_percentage * 100}%\n"
    )
    log_and_print(config_message)

    # Log the configuration details
    logging.info("Experiment Configuration:")
    logging.info(f"Database Sizes: {database_sizes}")
    logging.info(f"Number of Trials: {num_trials}")
    logging.info(f"Use Dynamic Epsilon: {use_dynamic_epsilon}")
    logging.info(f"Epsilon Adjustment Factors: {epsilon_adjust_factors}")
    logging.info(f"Target Percentage of Rows to Remove: {target_percentage * 100}%")

    # Start the experiment
    run_experiment(
        database_sizes=database_sizes,
        num_trials=num_trials,
        epsilon=epsilon,
        target_diff=target_diff,
        target_percentage=target_percentage,
        epsilon_adjust_factors=epsilon_adjust_factors,
        use_dynamic_epsilon=use_dynamic_epsilon
    )


if __name__ == "__main__":
    main()
