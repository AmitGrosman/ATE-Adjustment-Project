import os
import time
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration and Setup
 
logging.basicConfig(
    filename='experiment.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def log_and_print(message, level='info'):
    """Log message and print to console."""
    getattr(logging, level)(message)
    print(message)


# Data Loading and Preprocessing

def load_real_dataset(file_path):
    """Load and preprocess the dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")

    df = pd.read_csv(file_path)

    # Define formal education levels
    formal_education_levels = [
        "Bachelor's degree (BA, BS, B.Eng., etc.)",
        "Master's degree (MA, MS, M.Eng., MBA, etc.)",
        "Doctoral degree (PhD, EdD, etc.)"
    ]

    # Create treatment indicator 'T'
    df['T'] = df['FormalEducation'].isin(formal_education_levels).astype(int)
    df['O'] = df['ConvertedSalary']

    # Bin UndergradMajor
    df['UndergradMajor'] = df['UndergradMajor'].replace({
        'Computer science, computer engineering, or software engineering': 'CS & IT',
        'Information systems, information technology, or system administration': 'CS & IT',
        'Web development or web design': 'CS & IT',
        'Another engineering discipline (ex. civil, electrical, mechanical)': 'Engineering & Math',
        'Mathematics or statistics': 'Engineering & Math',
        'A natural science (ex. biology, chemistry, physics)': 'Other Disciplines',
        'A humanities discipline (ex. literature, history, philosophy)': 'Other Disciplines',
        'A business discipline (ex. accounting, finance, marketing)': 'Other Disciplines',
        'A social science (ex. anthropology, psychology, political science)': 'Other Disciplines',
        'Fine arts or performing arts (ex. graphic design, music, studio art)': 'Other Disciplines',
        'I never declared a major': 'Other Disciplines',
        'A health science (ex. nursing, pharmacy, radiology)': 'Other Disciplines'
    })

    # Bin RaceEthnicity
    df['RaceEthnicity'] = df['RaceEthnicity'].replace({
        'White or of European descent': 'European Descent',
        'South Asian': 'Asian Descent',
        'East Asian': 'Asian Descent',
        'Hispanic or Latino/Latina': 'Other',
        'Middle Eastern': 'Other',
        'Black or of African descent': 'Other',
        'I don’t know': 'Other',
        'I prefer not to say': 'Other',
        'Native American, Pacific Islander, or Indigenous Australian': 'Other',
        'Biracial': 'Other'
    })

    # Reorder and clean dataframe
    desired_columns = ['T', 'O'] + [col for col in df.columns if col not in ['T', 'O']]
    df = df[desired_columns].dropna(subset=['T', 'O'])

    return df


# ATE Calculation Utilities

def calculate_initial_aggregates(dataset, confounders):
    """Calculate sum of outcomes and counts for each subgroup."""
    grouped = dataset.groupby(['T'] + confounders)['O'].agg(['sum', 'count']).reset_index()
    aggregates = {}
    for _, row in grouped.iterrows():
        T = row['T']
        combo = tuple(row[conf] for conf in confounders)
        aggregates[(T, combo)] = {
            'sum_O': row['sum'],
            'count': row['count']
        }
    return aggregates


def calculate_expected_outcomes(aggregates, confounder_combinations, treatment):
    """Calculate expected outcome for a given treatment across all confounder combinations."""
    expected_Y = {}
    for combo in confounder_combinations:
        key = (treatment, combo)
        data = aggregates.get(key, {'sum_O': 0, 'count': 0})
        expected_Y[combo] = data['sum_O'] / data['count'] if data['count'] > 0 else 0
    return expected_Y

def calculate_probability_combinations(aggregates, confounder_combinations):
    """Calculate the probability of each confounder combination."""
    total_count = sum(
        aggregates.get((1, combo), {'count': 0})['count'] +
        aggregates.get((0, combo), {'count': 0})['count']
        for combo in confounder_combinations
    )
    if total_count == 0:
        return {combo: 0 for combo in confounder_combinations}

    return {
        combo: (
            aggregates.get((1, combo), {'count': 0})['count'] +
            aggregates.get((0, combo), {'count': 0})['count']
        ) / total_count
        for combo in confounder_combinations
    }

def calculate_ate_incremental(aggregates, confounder_combinations):
    """Calculate ATE using cached aggregates."""
    expected_Y_T1 = calculate_expected_outcomes(aggregates, confounder_combinations, treatment=1)
    expected_Y_T0 = calculate_expected_outcomes(aggregates, confounder_combinations, treatment=0)
    p_combo = calculate_probability_combinations(aggregates, confounder_combinations)

    ate = sum(
        p_combo[combo] * (expected_Y_T1[combo] - expected_Y_T0[combo])
        for combo in confounder_combinations
    )

    return ate

def calculate_ate_with_mask(dataset, mask, confounder_combinations, confounders):
    """Recalculate ATE based on a mask of included rows."""
    filtered_dataset = dataset[mask]
    aggregates = calculate_initial_aggregates(filtered_dataset, confounders)
    expected_Y_T1 = calculate_expected_outcomes(aggregates, confounder_combinations, treatment=1)
    expected_Y_T0 = calculate_expected_outcomes(aggregates, confounder_combinations, treatment=0)
    p_combo = calculate_probability_combinations(aggregates, confounder_combinations)

    ate = sum(
        p_combo[combo] * (expected_Y_T1[combo] - expected_Y_T0[combo])
        for combo in confounder_combinations
    )

    return ate

def calculate_ate_after_removal(row_idx, dataset, mask, confounder_combinations, confounders):
    """Calculate ATE as if a specific row was removed."""
    temp_mask = mask.copy()
    temp_mask[row_idx] = False
    return calculate_ate_with_mask(dataset, temp_mask, confounder_combinations, confounders)


# Candidate Evaluation Function

def evaluate_candidate(candidate, dataset, mask, confounder_combinations, confounders, target_ate):
    """Evaluate a candidate row removal and return the difference from target ATE."""
    T, combo, idx = candidate
    new_ate = calculate_ate_after_removal(idx, dataset, mask, confounder_combinations, confounders)
    diff = abs(new_ate - target_ate)
    return diff, idx, (T, combo)


# Removal algorithm
def per_group_removal_algorithm(dataset_info, target_ate, epsilon, confounder_combinations, confounders):
    dataset, aggregates = dataset_info
    mask = np.ones(len(dataset), dtype=bool)  # Initialize mask
    total_removals = 0

    removed_rows_details = []
    removals_by_group = defaultdict(int)

    iterations, differences = [], []

    # Precompute sorted indices within treatment and confounder groups
    sorted_indices = {}
    for T in [0, 1]:
        sorted_indices[T] = {}
        for combo in confounder_combinations:
            group_mask = (dataset['T'] == T) & (dataset[confounders] == combo).all(axis=1)
            sorted_group = dataset[group_mask].sort_values(
                by='O', ascending=(T == 0)  # Ascending for T=0, Descending for T=1
            )
            sorted_indices[T][combo] = sorted_group.index.to_numpy()

    current_ate = calculate_ate_with_mask(dataset, mask, confounder_combinations, confounders)

    while True:
        best_diff = float('inf')
        row_to_remove = None

        candidates = []
        for T in [0, 1]:
            for combo in confounder_combinations:
                available_indices = sorted_indices[T][combo][mask[sorted_indices[T][combo]]]
                if available_indices.size == 0:
                    continue

                # Adjust candidate selection based on current ATE vs target
                if current_ate < target_ate:
                    # To increase ATE, remove lowest 'O' from T=1 or highest 'O' from T=0
                    candidate_idx = available_indices[-1] if T == 1 else available_indices[0]
                else:
                    # To decrease ATE, remove highest 'O' from T=1 or lowest 'O' from T=0
                    candidate_idx = available_indices[0] if T == 1 else available_indices[-1]

                candidates.append((T, combo, candidate_idx))

        if not candidates:
            log_and_print("No more candidates available. Stopping.")
            break

        # Evaluate candidates to select the best one to remove
        for candidate in candidates:
            diff, idx, _ = evaluate_candidate(candidate, dataset, mask, confounder_combinations, confounders,
                                              target_ate)
            if diff < best_diff:
                best_diff = diff
                row_to_remove = idx

        if row_to_remove is None:
            log_and_print("No suitable row found for removal. Stopping.")
            break

        # Remove the selected row by updating the mask
        mask[row_to_remove] = False
        total_removals += 1

        # Track removed row details
        removed_row = dataset.iloc[row_to_remove]
        formal_education = 'Yes' if removed_row['T'] == 1 else 'No'
        group_details = {
            'Formal Education': formal_education,
            'Confounders': {conf: removed_row[conf] for conf in confounders}
        }
        group_key = ', '.join(f"{key}: {value}" for key, value in group_details['Confounders'].items())
        full_group_key = f"Formal Education: {formal_education}, {group_key}"
        removals_by_group[full_group_key] += 1

        row_details = {**{'FormalEducation': formal_education, 'Salary': removed_row['O']},
                       **removed_row[confounders].to_dict()}
        removed_rows_details.append(row_details)

        # Recalculate ATE using the updated mask
        current_ate = calculate_ate_with_mask(dataset, mask, confounder_combinations, confounders)
        iterations.append(total_removals)
        differences.append(current_ate)

        log_and_print(f"Total Removals: {total_removals}, ATE: {current_ate:.2f}, Difference: {best_diff:.2f}")

        # Check for convergence
        if abs(current_ate - target_ate) <= epsilon:
            log_and_print("Target ATE reached within epsilon. Stopping.")
            break

    average_salary = sum(row['Salary'] for row in removed_rows_details) / total_removals if total_removals > 0 else 0
    return total_removals, average_salary, iterations, differences, removed_rows_details, removals_by_group


# Experiment Runner
def run_experiment(file_path, desired_ate=None, epsilon=None, modify_current_ate=None, confounders=None):
    """Execute the ATE adjustment experiment."""
    log_and_print(f"\nLoading dataset from: {file_path}\n")

    dataset_df = load_real_dataset(file_path)

    missing_confounders = [conf for conf in confounders if conf not in dataset_df.columns]
    if missing_confounders:
        log_and_print(f"Confounders not found in dataset columns: {missing_confounders}", level='error')
        raise KeyError(f"Confounders not found in dataset columns: {missing_confounders}")

    log_and_print(f"Dataset loaded successfully with {len(dataset_df)} records.")
    log_and_print(f"Confounders: {confounders}")

    aggregates = calculate_initial_aggregates(dataset_df, confounders)
    confounder_combinations = list(dataset_df.groupby(confounders).size().index)
    dataset_info = (dataset_df, aggregates)

    if modify_current_ate is not None:
        current_ate = calculate_ate_incremental(aggregates, confounder_combinations)
        target_ate = current_ate + modify_current_ate
        log_and_print(f"Current ATE: {current_ate:.2f}, Modified ATE: {target_ate:.2f}, Epsilon: {epsilon}")
    elif desired_ate is not None and epsilon is not None:
        current_ate = calculate_ate_incremental(aggregates, confounder_combinations)
        log_and_print(f"Current ATE: {current_ate:.2f}, Desired ATE: {desired_ate:.2f}, Epsilon: {epsilon}")
        target_ate = desired_ate
    else:
        log_and_print("Error: Provide either desired_ate or modify_current_ate with epsilon.", level='error')
        return

    log_and_print(f"Confounder Combinations: {len(confounder_combinations)}")

    start_time = time.time()
    results = per_group_removal_algorithm(
        dataset_info, target_ate, epsilon, confounder_combinations, confounders
    )
    elapsed_time = time.time() - start_time

    total_removals, average_salary, iterations, differences, removed_rows_details, removals_by_group = results

    log_and_print("\n" + "=" * 50 + "\n")
    log_and_print(f"Experiment completed in {elapsed_time:.2f} seconds.")
    log_and_print(f"Total removals: {total_removals}")
    log_and_print(f"Average salary of removed rows: {average_salary:.2f}")

    log_and_print("\nGroups from which rows were removed:")
    for group, count in removals_by_group.items():
        log_and_print(f"  {group}: {count} rows removed")

    log_and_print("\nDetails of Removed Rows (In Order):")
    for row in removed_rows_details:
        log_and_print(f"  {row}")

    # Plot Iterations vs. ATE Differences
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, differences, marker='o', linestyle='-')
    plt.title('Iterations vs. ATE Differences')
    plt.xlabel('Iterations')
    plt.ylabel('ATE')
    plt.axhline(y=target_ate, color='r', linestyle='--', label='Target ATE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('iterations_vs_ATE.png')
    plt.show()


# Main Execution
def main():
    """Configure and run the experiment."""
    # Configuration
    file_path = 'stackoverflow_db.csv'
    confounders = ['UndergradMajor', 'Continent', 'RaceEthnicity']

    # Experiment Settings
    desired_ate = None  # Option 1: Set a specific desired ATE
    modify_current_ate = 500  # Option 2: Adjust current ATE by this amount
    epsilon = 1  # Acceptable deviation from target ATE

    log_and_print("Starting ATE adjustment experiment.")

    # Run the experiment
    run_experiment(
        file_path=file_path,
        desired_ate=desired_ate,
        epsilon=epsilon,
        modify_current_ate=modify_current_ate,
        confounders=confounders
    )

    log_and_print("\nExperiment completed successfully.")

if __name__ == "__main__":
    main()
