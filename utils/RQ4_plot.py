import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages


OLD_CLASSES = ['CWE-119', 'CWE-125', 'CWE-190', 'CWE-20', 'CWE-200', 
               'CWE-399', 'CWE-416', 'CWE-476', 'CWE-703', 'CWE-787']


CLASSES = ['CWE-125', 'CWE-119', 'CWE-787', 'CWE-20', 'CWE-416', 
           'CWE-476', 'CWE-703', 'CWE-200', 'CWE-190', 'CWE-399']

CM_GEN = np.array([
    [ 95,  8,  2,  5,  6,  5,  4,  7,  5,  7],
    [  8,125,  5,  5,  1,  0,  2, 10,  5,  3],
    [  8,  6, 39,  4,  2,  2,  1,  3,  0,  2],
    [ 13,  4,  2, 90,  6,  4,  1,  8,  3,  1],
    [  6,  3,  1,  7, 44,  1,  4,  3,  1,  1],
    [  7,  0,  1,  5,  6, 20,  0,  2,  4,  1],
    [ 12,  3,  0,  7,  7,  2, 57,  8,  3,  1],
    [ 12,  7,  0,  7,  9,  0,  3, 46,  6,  2],
    [ 11, 10,  2,  7,  4,  0,  0,  7, 28,  4],
    [ 16, 16,  2,  6, 11,  1,  3,  8,  3, 72]
])

CM_DISC = np.array([
    [ 98,  9,  1,  7,  1,  3,  5,  6,  7,  7],
    [ 11,121,  5,  4,  1,  0,  3, 10,  1,  8],
    [  7, 10, 35,  4,  3,  0,  3,  0,  4,  1],
    [ 21,  5,  2, 72,  4,  2,  3,  9,  8,  6],
    [  5,  4,  3, 14, 31,  1,  3,  2,  6,  2],
    [  6,  2,  0,  5,  5, 14,  5,  4,  2,  3],
    [  2, 11,  1,  9,  2,  2, 57, 13,  1,  2],
    [  7,  6,  1,  7,  2,  3,  7, 49,  7,  3],
    [  7, 11,  2,  9,  2,  1,  4,  7, 24,  6],
    [ 15, 19,  1,  4,  1,  1,  2,  8,  4, 83]
])


idx = [OLD_CLASSES.index(c) for c in CLASSES]
CM_DISC = CM_DISC[np.ix_(idx, idx)]
CM_GEN = CM_GEN[np.ix_(idx, idx)]


def plot_cm_to_pdf(pdf, cm, method_type):
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            pct = cm_norm[i, j] * 100
            cnt = cm[i, j]
            row.append(f'{pct:.1f}%\n({cnt})')
        annotations.append(row)
    
    fig, ax = plt.subplots(figsize=(14, 11))
    cmap = 'Blues' if method_type == 'discriminative' else 'Greens'
    
    sns.heatmap(cm_norm, annot=annotations, fmt='', cmap=cmap,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                annot_kws={'fontsize': 16},vmin=0, vmax=1.0)
    
    ax.set_xlabel('Predicted CWE', fontsize=16, fontweight='bold')
    ax.set_ylabel('True CWE', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', rotation=0, labelsize=14)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    output_dir = os.path.join('results', 'RQ4', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'confusion_matrices.pdf')
    
    with PdfPages(output_path) as pdf:
        plot_cm_to_pdf(pdf, CM_DISC, 'discriminative')
        plot_cm_to_pdf(pdf, CM_GEN, 'generative')
        
    print(f"Single PDF generated in: {output_path}")