import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

models = [
    {'name': 'CodeBERT', 'folder': 'codebert'},
    {'name': 'GraphCodeBERT', 'folder': 'graphcodebert'},
    {'name': 'UnixCoder', 'folder': 'unixcoder'},
]
datasets = ['devign', 'reveal', 'bigvul']
dataset_display = {'devign': 'Devign', 'reveal': 'Reveal', 'bigvul': 'Big-Vul'}

jpg_path = r"results\RQ1\venn.jpg"
pdf_path = r"results\RQ1\venn.pdf"

fig_grid, axes_grid = plt.subplots(len(models), len(datasets),
                                   figsize=(5 * len(datasets), 5 * len(models)))
if len(models) == 1:
    axes_grid = [axes_grid]
if len(datasets) == 1:
    axes_grid = [[ax] for ax in axes_grid]

for i, model in enumerate(models):
    for j, ds in enumerate(datasets):
        ax_grid = axes_grid[i][j]
        path_gen = rf"results\RQ1\{model['folder']}\generate\{ds}\predictions.csv"
        path_cls = rf"results\RQ1\{model['folder']}\classify\{ds}\predictions.csv"

        try:
            df_gen = pd.read_csv(path_gen)
            df_cls = pd.read_csv(path_cls)
            tp_gen = set(df_gen[(df_gen['prediction'] == 1) & (df_gen['true_label'] == 1)]['idx'])
            tp_cls = set(df_cls[(df_cls['prediction'] == 1) & (df_cls['true_label'] == 1)]['idx'])
        except FileNotFoundError:
            tp_gen = set()
            tp_cls = set()

        v_grid = venn2([tp_cls, tp_gen],
                       set_labels=('Disc', 'Gen'),
                       set_colors=('#9E86E3', '#1496D4'),
                       ax=ax_grid)
        if v_grid:
            for text in v_grid.set_labels:
                if text:
                    text.set_fontsize(18)
                    # text.set_fontweight('bold')
                    x, y = text.get_position()
                    text.set_position((x, y - 0.02))
            for text in v_grid.subset_labels:
                if text:
                    text.set_fontsize(20)
                    text.set_fontweight('medium')

        ax_grid.set_title(f"{model['name']} - {dataset_display[ds]}",
                          fontsize=20, fontweight='medium',
                          y=0.94)

fig_grid.subplots_adjust(wspace=0.15, hspace=0.2)
fig_grid.tight_layout()
fig_grid.savefig(jpg_path, bbox_inches='tight', dpi=300)
fig_grid.savefig(pdf_path, bbox_inches='tight')
plt.close(fig_grid)