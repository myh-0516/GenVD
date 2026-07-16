import json
import os
import csv
from transformers import AutoTokenizer

DATASET_CONFIG = {
    'devign': {'code_field': 'func', 'label_field': 'target', 'id_field': 'idx'},
    'bigvul': {'code_field': 'func', 'label_field': 'target', 'id_field': 'idx'},
    'reveal': {'code_field': 'functionSource', 'label_field': 'label', 'id_field': 'hash'},
    'diversevul': {'code_field': 'func', 'label_field': 'target', 'id_field': 'idx'}
}

DATASETS = ['diversevul'] #'devign', 'bigvul', 'reveal', 'diversevul'
SPLITS = ['train', 'valid', 'test']

tokenizer = AutoTokenizer.from_pretrained("pretrained_models\codebert-base")
tokenizer.model_max_length = int(1e9)

def fmt(x):
    return f"{x * 100:.2f}%"

os.makedirs('datasets/statistics', exist_ok=True)

summary_rows = []

for dataset in DATASETS:
    config = DATASET_CONFIG[dataset]
    code_field = config['code_field']
    label_field = config['label_field']
    id_field = config['id_field']

    dataset_stat_dir = f'datasets/{dataset}/statistics'
    os.makedirs(dataset_stat_dir, exist_ok=True)

    overall = {
        'total': 0,
        'len_sum': 0,
        'max_len': 0,
        'over_512': 0,
        'vul_total': 0,
        'vul_len_sum': 0,
        'vul_over_512': 0,
        'non_total': 0,
        'non_len_sum': 0,
        'non_over_512': 0
    }

    for split in SPLITS:
        path = f'datasets/{dataset}/{split}.jsonl'
        if not os.path.exists(path):
            continue

        stats = {
            'total': 0,
            'len_sum': 0,
            'max_len': 0,
            'over_512': 0,
            'vul_total': 0,
            'vul_len_sum': 0,
            'vul_over_512': 0,
            'non_total': 0,
            'non_len_sum': 0,
            'non_over_512': 0
        }

        sample_rows = []

        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                code = data.get(code_field, "")
                label = int(data.get(label_field, 0))

                token_len = len(tokenizer.tokenize(code))
                is_truncated = token_len > 512

                if id_field and id_field in data:
                    sample_id = data[id_field]
                else:
                    sample_id = f"{dataset}_{split}_{idx}"

                sample_rows.append([sample_id, label, token_len, int(is_truncated)])

                stats['total'] += 1
                stats['len_sum'] += token_len
                stats['max_len'] = max(stats['max_len'], token_len)
                if is_truncated:
                    stats['over_512'] += 1

                if label == 1:
                    stats['vul_total'] += 1
                    stats['vul_len_sum'] += token_len
                    if is_truncated:
                        stats['vul_over_512'] += 1
                else:
                    stats['non_total'] += 1
                    stats['non_len_sum'] += token_len
                    if is_truncated:
                        stats['non_over_512'] += 1

        with open(os.path.join(dataset_stat_dir, f'{split}.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'label', 'token_len', 'is_truncated'])
            writer.writerows(sample_rows)

        summary_rows.append([
            dataset, split,
            stats['total'],
            round(stats['len_sum'] / stats['total'], 2) if stats['total'] else 0,
            stats['max_len'],
            stats['over_512'],
            fmt(stats['over_512'] / stats['total']) if stats['total'] else "0.00%",
            stats['vul_total'],
            round(stats['vul_len_sum'] / stats['vul_total'], 2) if stats['vul_total'] else 0,
            stats['vul_over_512'],
            fmt(stats['vul_over_512'] / stats['vul_total']) if stats['vul_total'] else "0.00%",
            stats['non_total'],
            round(stats['non_len_sum'] / stats['non_total'], 2) if stats['non_total'] else 0,
            stats['non_over_512'],
            fmt(stats['non_over_512'] / stats['non_total']) if stats['non_total'] else "0.00%"
        ])

        for k in overall:
            overall[k] += stats[k]

    if overall['total'] > 0:
        summary_rows.append([
            dataset, 'overall',
            overall['total'],
            round(overall['len_sum'] / overall['total'], 2),
            overall['max_len'],
            overall['over_512'],
            fmt(overall['over_512'] / overall['total']),
            overall['vul_total'],
            round(overall['vul_len_sum'] / overall['vul_total'], 2) if overall['vul_total'] else 0,
            overall['vul_over_512'],
            fmt(overall['vul_over_512'] / overall['vul_total']) if overall['vul_total'] else "0.00%",
            overall['non_total'],
            round(overall['non_len_sum'] / overall['non_total'], 2) if overall['non_total'] else 0,
            overall['non_over_512'],
            fmt(overall['non_over_512'] / overall['non_total']) if overall['non_total'] else "0.00%"
        ])

with open('datasets/statistics/length_stats.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'dataset','split','total','avg_len','max_len','over_512','over_512_ratio',
        'vul_total','vul_avg_len','vul_over_512','vul_over_512_ratio',
        'non_total','non_avg_len','non_over_512','non_over_512_ratio'
    ])
    writer.writerows(summary_rows)

print("Saved to datasets/statistics/length_stats.csv")