import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

input_file = os.path.join('datasets', 'diversevul', 'diversevul_20230702.json')


output_dir = os.path.join('datasets', 'diversevul_cwe2')

os.makedirs(output_dir, exist_ok=True)

data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

df['cwe_label'] = df['cwe'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
df = df[(df['target'] == 1) & (df['cwe_label'].notna())]

# cwe_counts = df['cwe_label'].value_counts().reset_index()
# cwe_counts.columns = ['CWE', 'Count']
# cwe_counts.to_csv(os.path.join(output_dir, 'cwe_statistics.csv'), index=False, encoding='utf-8')

cwe_counts_top10 = df['cwe_label'].value_counts().nlargest(10).reset_index()
cwe_counts_top10.columns = ['CWE', 'Count']
cwe_counts_top10.to_csv(os.path.join(output_dir, 'cwe_top10_statistics.csv'), index=False, encoding='utf-8')

top_10_cwes = df['cwe_label'].value_counts().nlargest(10).index.tolist()
df_top10 = df[df['cwe_label'].isin(top_10_cwes)]

df_final = df_top10[['hash', 'func', 'target', 'cwe_label']]

train_df, temp_df = train_test_split(df_final, test_size=0.2, random_state=42, stratify=df_final['cwe_label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['cwe_label'])

train_df.to_json(os.path.join(output_dir, 'train.jsonl'), orient='records', lines=True, force_ascii=False)
val_df.to_json(os.path.join(output_dir, 'valid.jsonl'), orient='records', lines=True, force_ascii=False)
test_df.to_json(os.path.join(output_dir, 'test.jsonl'), orient='records', lines=True, force_ascii=False)