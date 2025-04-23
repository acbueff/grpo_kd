# src/analyze_results.py
import argparse
import os
import json
import glob
import logging
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_results(results_dir: str) -> pd.DataFrame:
    """Parses evaluation results from JSON files into a pandas DataFrame."""
    summary_path = os.path.join(results_dir, "evaluation_summary.json")
    if not os.path.exists(summary_path):
        logger.error(f"Evaluation summary file not found: {summary_path}")
        return pd.DataFrame() # Return empty DataFrame

    try:
        with open(summary_path, 'r') as f:
            all_results_dict = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {summary_path}: {e}")
        return pd.DataFrame()

    parsed_data = []
    for run_name, datasets_results in all_results_dict.items():
        if "error" in datasets_results:
            logger.warning(f"Skipping run {run_name} due to evaluation error: {datasets_results['error']}")
            continue

        try:
            # Extract model type and seed from run_name (e.g., "baseline_seed0")
            parts = run_name.split('_seed')
            if len(parts) != 2:
                logger.warning(f"Could not parse model type/seed from run name: {run_name}. Skipping.")
                continue
            model_type = parts[0]
            seed = int(parts[1])

            for dataset_name, metrics_results in datasets_results.items():
                if not isinstance(metrics_results, dict):
                    logger.warning(f"Unexpected format for dataset results '{dataset_name}' in run '{run_name}'. Skipping.")
                    continue
                num_samples = metrics_results.get("num_samples")
                for metric_name, value in metrics_results.items():
                    if metric_name != "num_samples" and value is not None: # Exclude None values
                         # Ensure value is float or int for stats
                         try:
                             metric_value = float(value)
                             parsed_data.append({
                                 "model_type": model_type,
                                 "seed": seed,
                                 "dataset": dataset_name,
                                 "metric": metric_name,
                                 "value": metric_value,
                                 "num_samples": num_samples
                             })
                         except (ValueError, TypeError):
                              logger.warning(f"Could not convert metric '{metric_name}' value '{value}' to float for run '{run_name}', dataset '{dataset_name}'. Skipping.")

        except Exception as e:
             logger.error(f"Error processing results for run {run_name}: {e}", exc_info=True)


    if not parsed_data:
         logger.warning("No valid data parsed from the results file.")
         return pd.DataFrame()

    df = pd.DataFrame(parsed_data)
    logger.info(f"Successfully parsed results into DataFrame with {len(df)} rows.")
    # logger.info(f"DataFrame head:\n{df.head()}")
    return df

def bootstrap_ci(data, n_bootstrap=1000, ci_level=0.95):
    """Calculate bootstrap confidence interval for the mean."""
    data = np.asarray(data) # Ensure input is numpy array
    if len(data) < 2 or np.all(np.isnan(data)): return np.nan, np.nan # Need at least 2 points for CI
    valid_data = data[~np.isnan(data)] # Remove NaNs for bootstrapping
    if len(valid_data) < 2: return np.nan, np.nan

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(valid_data, size=len(valid_data), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower_percentile = (1 - ci_level) / 2 * 100
    upper_percentile = (1 + ci_level) / 2 * 100
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    return ci_lower, ci_upper

def analyze(df: pd.DataFrame, alpha: float = 0.05):
    """Performs statistical analysis on the parsed results DataFrame."""
    if df.empty:
        logger.error("Cannot perform analysis on empty DataFrame.")
        return

    analysis_results = defaultdict(lambda: defaultdict(dict)) # Nested defaultdict
    report_lines = ["# Statistical Analysis Report", "="*30, ""]

    # Group by dataset and metric
    grouped = df.groupby(['dataset', 'metric'])

    for name, group in grouped:
        dataset, metric = name
        logger.debug(f"Analyzing: Dataset={dataset}, Metric={metric}")
        report_lines.append(f"## Dataset: {dataset} | Metric: {metric}")

        # Pivot table for easier access to baseline/dappo pairs per seed
        try:
            pivot = group.pivot_table(index='seed', columns='model_type', values='value')
            baseline_results = pivot['baseline'].dropna().values
            dappo_results = pivot['da-ppo'].dropna().values
        except KeyError as e:
            logger.warning(f"Could not create pivot table for {dataset}/{metric} (missing 'baseline' or 'da-ppo' column? Error: {e}). Skipping analysis.")
            report_lines.append(f"  - Error: Missing data for baseline or da-ppo.")
            report_lines.append("")
            continue
        except Exception as e:
            logger.error(f"Unexpected error creating pivot table for {dataset}/{metric}: {e}. Skipping analysis.")
            report_lines.append(f"  - Error: Could not process data for analysis.")
            report_lines.append("")
            continue


        # Check if we have results for both types
        if len(baseline_results) == 0 or len(dappo_results) == 0:
            logger.warning(f"Missing results for baseline or da-ppo for {dataset}/{metric} after pivoting/dropna. Skipping analysis.")
            report_lines.append("  - Missing data for one or both model types.")
            report_lines.append("")
            continue

        # --- Descriptive Stats & CIs ---
        mean_baseline = np.mean(baseline_results)
        std_baseline = np.std(baseline_results)
        ci_low_base, ci_high_base = bootstrap_ci(baseline_results, ci_level=(1-alpha))
        report_lines.append(f"  - Baseline PPO : Mean={mean_baseline:.4f}, Std={std_baseline:.4f}, {int((1-alpha)*100)}% CI=[{ci_low_base:.4f}, {ci_high_base:.4f}] (n={len(baseline_results)})" if pd.notna(ci_low_base) else f"  - Baseline PPO : Mean={mean_baseline:.4f}, Std={std_baseline:.4f}, CI=N/A (n={len(baseline_results)})")

        mean_dappo = np.mean(dappo_results)
        std_dappo = np.std(dappo_results)
        ci_low_dappo, ci_high_dappo = bootstrap_ci(dappo_results, ci_level=(1-alpha))
        report_lines.append(f"  - DA-PPO       : Mean={mean_dappo:.4f}, Std={std_dappo:.4f}, {int((1-alpha)*100)}% CI=[{ci_low_dappo:.4f}, {ci_high_dappo:.4f}] (n={len(dappo_results)})" if pd.notna(ci_low_dappo) else f"  - DA-PPO       : Mean={mean_dappo:.4f}, Std={std_dappo:.4f}, CI=N/A (n={len(dappo_results)})")

        analysis_results[dataset][metric] = {
             "baseline_mean": mean_baseline, "baseline_std": std_baseline, "baseline_ci": (ci_low_base, ci_high_base), "baseline_n": len(baseline_results),
             "dappo_mean": mean_dappo, "dappo_std": std_dappo, "dappo_ci": (ci_low_dappo, ci_high_dappo), "dappo_n": len(dappo_results),
        }

        # --- Paired Comparison (use aligned data from pivot) ---
        # Get paired data ensuring alignment by seed index from pivot
        aligned_data = pivot[['baseline', 'da-ppo']].dropna()
        if len(aligned_data) < 2: # Need at least 2 pairs
             logger.warning(f"Not enough paired data points ({len(aligned_data)}) for paired test on {dataset}/{metric} after alignment.")
             report_lines.append("  - Paired Test: Not enough valid pairs after alignment.")
        else:
             baseline_aligned = aligned_data['baseline'].values
             dappo_aligned = aligned_data['da-ppo'].values
             diffs = dappo_aligned - baseline_aligned

             mean_diff = np.mean(diffs)
             std_diff = np.std(diffs)
             ci_low_diff, ci_high_diff = bootstrap_ci(diffs, ci_level=(1-alpha))
             report_lines.append(f"  - Mean Difference (DA-PPO - Baseline): {mean_diff:.4f}, Std={std_diff:.4f}, {int((1-alpha)*100)}% CI=[{ci_low_diff:.4f}, {ci_high_diff:.4f}] (n={len(diffs)})" if pd.notna(ci_low_diff) else f"  - Mean Difference (DA-PPO - Baseline): {mean_diff:.4f}, Std={std_diff:.4f}, CI=N/A (n={len(diffs)})")

             # Perform paired t-test
             try:
                 t_stat, p_val_ttest = stats.ttest_rel(dappo_aligned, baseline_aligned)
                 report_lines.append(f"  - Paired t-test : p-value = {p_val_ttest:.4f}")
             except Exception as e:
                  logger.error(f"Paired t-test failed for {dataset}/{metric}: {e}")
                  p_val_ttest = np.nan
                  report_lines.append(f"  - Paired t-test : FAILED")


             # Perform Wilcoxon signed-rank test (non-parametric)
             try:
                 # Wilcoxon needs > 0 differences and typically > ~7-8 samples for good p-value approx.
                 zero_diff = np.all(np.abs(diffs) < 1e-9)
                 if zero_diff:
                     stat_wilcoxon, p_val_wilcoxon = np.nan, 1.0 # No difference
                     logger.debug("Wilcoxon: All differences are zero.")
                 elif len(diffs) < 8: # Small sample size warning
                     logger.warning(f"Small sample size ({len(diffs)}) for Wilcoxon test on {dataset}/{metric}. P-value might be approximate.")
                     stat_wilcoxon, p_val_wilcoxon = stats.wilcoxon(diffs, zero_method='pratt') # Pratt handles zeros
                 else:
                     stat_wilcoxon, p_val_wilcoxon = stats.wilcoxon(diffs, zero_method='pratt')
                 report_lines.append(f"  - Wilcoxon test : p-value = {p_val_wilcoxon:.4f}" if pd.notna(p_val_wilcoxon) else "  - Wilcoxon test : N/A (all diffs zero?)")
             except ValueError as e:
                  logger.warning(f"Wilcoxon test failed for {dataset}/{metric} (perhaps all differences are zero or issue with data?): {e}")
                  stat_wilcoxon, p_val_wilcoxon = np.nan, np.nan # Indicate test could not be run
                  report_lines.append(f"  - Wilcoxon test : FAILED ({e})")


             significant_ttest = pd.notna(p_val_ttest) and p_val_ttest < alpha
             significant_wilcoxon = pd.notna(p_val_wilcoxon) and p_val_wilcoxon < alpha

             # Decide significance based on either test passing
             is_significant = significant_ttest or significant_wilcoxon
             analysis_results[dataset][metric]["significant"] = is_significant
             analysis_results[dataset][metric]["winner"] = "None"

             if is_significant:
                  # Determine winner based on mean difference sign
                  # Assumes higher metric value is better. Need to adjust if lower is better (e.g., perplexity)
                  higher_is_better = not metric.lower() in ["perplexity", "loss", "error_rate", "ger"] # Add other lower-is-better metrics
                  if mean_diff > 0 and higher_is_better:
                      winner = "DA-PPO"
                  elif mean_diff < 0 and not higher_is_better:
                      winner = "DA-PPO"
                  elif mean_diff < 0 and higher_is_better:
                      winner = "Baseline PPO"
                  elif mean_diff > 0 and not higher_is_better:
                      winner = "Baseline PPO"
                  else: # mean_diff is zero
                      winner = "None"

                  report_lines.append(f"  - RESULT: Statistically Significant difference found (p < {alpha}). Winner: {winner}")
                  analysis_results[dataset][metric]["winner"] = winner
             else:
                  report_lines.append(f"  - RESULT: No statistically significant difference found (p >= {alpha}).")

             analysis_results[dataset][metric]["paired_t_pvalue"] = p_val_ttest
             analysis_results[dataset][metric]["wilcoxon_pvalue"] = p_val_wilcoxon
             analysis_results[dataset][metric]["mean_diff"] = mean_diff
             analysis_results[dataset][metric]["mean_diff_ci"] = (ci_low_diff, ci_high_diff)

        report_lines.append("") # Add space between metrics

    # --- Print Report ---
    print("\n".join(report_lines))

    # --- Save Report ---
    report_path = os.path.join(results_dir, "statistical_analysis_report.md") # Save in results dir
    try:
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        logger.info(f"Statistical analysis report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save analysis report: {e}")

    # --- Save Parsed Analysis Data (Optional) ---
    analysis_summary_path = os.path.join(results_dir, "analysis_summary.json") # Save in results dir
    try:
         # Convert defaultdict to dict for JSON serialization
         analysis_results_dict = json.loads(json.dumps(analysis_results, default=lambda x: None if pd.isna(x) else x)) # Handle potential NaNs
         with open(analysis_summary_path, 'w') as f:
             json.dump(analysis_results_dict, f, indent=2)
         logger.info(f"Analysis summary data saved to: {analysis_summary_path}")
    except Exception as e:
         logger.error(f"Failed to save analysis summary JSON: {e}")


def main(results_dir: str, config_path: str):
    """Main function to run analysis."""

    # --- Load Config for settings ---
    logger.info(f"Loading configuration from: {config_path}")
    alpha = 0.05 # Default significance level
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        alpha = config.get('significance_level', 0.05)
        logger.info(f"Using significance level (alpha): {alpha}")
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}. Using default alpha={alpha}. Ensure the config path is correct.")
        # Allow proceeding with default alpha
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}. Using default alpha={alpha}.")


    # --- Parse Results ---
    results_df = parse_results(results_dir)

    # --- Perform Analysis ---
    if not results_df.empty:
        analyze(results_df, alpha=alpha)
    else:
        logger.error("Analysis aborted due to errors in parsing results or no results found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze evaluation results and perform statistical tests.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing the 'evaluation_summary.json' file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the main configuration YAML file (to get alpha). Note: Should be the same config used for training/evaluation.")
    args = parser.parse_args()

    main(args.results_dir, args.config)
