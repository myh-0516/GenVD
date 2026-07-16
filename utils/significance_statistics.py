import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

def cliffs_delta(x, y):
    n, m = len(x), len(y)
    mat = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if x[i] > y[j]:
                mat[i, j] = 1
            elif x[i] < y[j]:
                mat[i, j] = -1
    return np.sum(mat) / (n * m)

def cliffs_delta_ci(x, y, alpha=0.05, n_bootstraps=10000):
    bootstrapped_deltas = []
    np.random.seed(42)
    
    for _ in range(n_bootstraps):
        x_boot = np.random.choice(x, size=len(x), replace=True)
        y_boot = np.random.choice(y, size=len(y), replace=True)
        bootstrapped_deltas.append(cliffs_delta(x_boot, y_boot))
        
    bootstrapped_deltas = np.sort(bootstrapped_deltas)
    lower_bound = np.percentile(bootstrapped_deltas, (alpha / 2) * 100)
    upper_bound = np.percentile(bootstrapped_deltas, (1 - alpha / 2) * 100)
    
    return lower_bound, upper_bound

def compute_metrics(x, y, model, dataset, metric, n_pairs):
    if np.all(x == y):
        p_value = 1.0
    else:
        try:
            _, p_value = stats.wilcoxon(x, y)
        except ValueError:
            p_value = float('nan')
            
    delta = cliffs_delta(x, y)
    ci_lower, ci_upper = cliffs_delta_ci(x, y)
    
    if abs(delta) < 0.147: size = "Negligible"
    elif abs(delta) < 0.33: size = "Small"
    elif abs(delta) < 0.474: size = "Medium"
    else: size = "Large"
    
    mean_x = np.mean(x)
    std_x = np.std(x, ddof=1) if len(x) > 1 else 0
    mean_y = np.mean(y)
    std_y = np.std(y, ddof=1) if len(y) > 1 else 0
    
    return {
        'Model': model,
        'Dataset': dataset,
        'Metric': metric,
        'Pairs': n_pairs,
        'Gen_Mean±Std': f"{mean_x:.4f}±{std_x:.4f}",
        'Cls_Mean±Std': f"{mean_y:.4f}±{std_y:.4f}",
        'Wilcoxon_p': round(p_value, 5) if pd.notna(p_value) else 'N/A',
        'Cliffs_Delta': round(delta, 4),
        'Effect_Size': size,
        '95%_CI': f"[{ci_lower:.4f}, {ci_upper:.4f}]"
    }

def analyze_results(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    target_metrics = ['f1_binary']
    # target_metrics = ['accuracy', 'precision', 'recall', 'f1_binary']
    
    results = []
    for (model, dataset), group in df.groupby(['model', 'dataset']):
        merged = pd.merge(group[group['method'] == 'gen'], group[group['method'] == 'cls'], on='seed', suffixes=('_gen', '_cls'))
        if len(merged) == 0:
            continue
        for metric in target_metrics:
            x = merged[f'{metric}_gen'].values
            y = merged[f'{metric}_cls'].values
            results.append(compute_metrics(x, y, model, dataset, metric, len(merged)))
            
    pd.DataFrame(results).to_csv(output_csv, index=False)

if __name__ == "__main__":
    INPUT_FILE = r"results\RQ1\all_results.csv"
    OUTPUT_FILE = r"results\RQ1\statistic_f1.csv"
    analyze_results(INPUT_FILE, OUTPUT_FILE)