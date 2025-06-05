import os
import re
import numpy as np

def extract_f1_score(log_file):
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            match = re.search(r'Average F-score ([0-9.]+)%', content)
            if match:
                return float(match.group(1)) / 100
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    return None

def calculate_avg_score_new_structure(model_type, dataset):
    """Calculate scores for new directory structure: log/{model_type}-{dataset}-split{split}/log_train.txt"""
    scores = []
    split_scores = {}
    for split in range(5):
        log_path = f"log/{model_type}-{dataset}-split{split}/log_train.txt"
        if os.path.exists(log_path):
            score = extract_f1_score(log_path)
            if score is not None:
                scores.append(score)
                split_scores[split] = score
                
    if scores:
        return np.mean(scores), np.std(scores), split_scores
    return None, None, {}

# Initialize results dictionary
results = {}

# Detect all model directories in log folder
model_dirs = set()
for item in os.listdir("log"):
    if os.path.isdir(os.path.join("log", item)):
        # Parse model name from directory structure
        parts = item.split("-")
        if len(parts) >= 3 and parts[-2] in ["summe", "tvsum"] and parts[-1].startswith("split"):
            # This is likely a model directory with new structure
            model_name = "-".join(parts[:-2])
            model_dirs.add(model_name)

# For the new directory structure (all detected models)
for model_type in sorted(list(model_dirs)):
    for dataset in ['summe', 'tvsum']:
        mean, std, individual_scores = calculate_avg_score_new_structure(model_type, dataset)
        if mean is not None:
            results[f"{model_type}-{dataset}"] = (mean, std, individual_scores)

# Prepare results for display and saving
output_lines = []
output_lines.append("\n" + "="*80)
output_lines.append("AVERAGE F1-SCORES")
output_lines.append("="*80)
output_lines.append(f"{'Model':<25} {'SumMe':<25} {'TVSum':<25}")
output_lines.append("-"*80)

# Separate models into supervised and unsupervised
supervised_models = []
unsupervised_models = []

for model_key in results.keys():
    model = model_key.split('-')[0]  # Get model type without dataset
    model_full = model_key.split('-' + model_key.split('-')[-1])[0]  # Full model name without dataset
    
    if "DSNsup" in model_key:
        if model_full not in supervised_models:
            supervised_models.append(model_full)
    else:
        if model_full not in unsupervised_models:
            unsupervised_models.append(model_full)

# Sort model lists
supervised_models.sort()
unsupervised_models.sort()

# Helper function to format results
def format_result(model, dataset):
    result = results.get(f'{model}-{dataset}', (None, None, {}))
    if result[0] is None:
        return "N/A"
    
    mean, std, split_scores = result
    split_strs = []
    for split in sorted(split_scores.keys()):
        split_strs.append(f"S{split}:{split_scores[split]:.1%}")
    score_str = ", ".join(split_strs)
    return f"{mean:.1%} ± {std:.1%} [{score_str}]"

# Add supervised models results
output_lines.append("SUPERVISED MODELS:")
for model in supervised_models:
    summe_res = format_result(model, 'summe')
    tvsum_res = format_result(model, 'tvsum')
    output_lines.append(f"{model:<25} {summe_res:<25} {tvsum_res:<25}")

output_lines.append("-"*80)

# Add unsupervised models results
output_lines.append("UNSUPERVISED MODELS:")
for model in unsupervised_models:
    summe_res = format_result(model, 'summe')
    tvsum_res = format_result(model, 'tvsum')
    output_lines.append(f"{model:<25} {summe_res:<25} {tvsum_res:<25}")

output_lines.append("="*80)

# Print results to console
for line in output_lines:
    print(line)

# Save results to file
with open("model_scores.txt", "w") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"\nResults have been saved to model_scores.txt")
