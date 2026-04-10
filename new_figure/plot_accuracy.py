"""
KDD - (MNIST / FMNIST / CIFAR-10)
- 7pt
- figsize=(7.0, 2.2), 1×3 , ,
- Legend ncol=4
- , bbox_inches='tight'
- PDF new_figure/
"""
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── ──
rcParams.update({
    'font.size': 7,
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUTPUT_DIR = SCRIPT_DIR


def parse_training_log(log_file):
    """Parse training log to extract test accuracy per round."""
    if not os.path.exists(log_file):
        return None
    test_acc = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(
                r'Round\s+\d+:\s+Time=[\d.]+s,\s+Loss=[\d.]+,\s+Test Acc=([\d.]+)%',
                line)
            if m:
                test_acc.append(float(m.group(1)))
    return test_acc if test_acc else None


def load_dataset_histories(dataset_name):
    """Load all four methods' histories for a dataset."""
    method_files = {
        'SplitFed':          'SplitFed',
        'Multi-Tenant SFL':  'Multi-Tenant_SFL',
        'COIN-UCB':          'LENS-UCB',
        'Full-Info':         'Full-Info',
    }
    histories = {}
    for display_name, file_name in method_files.items():
        path = os.path.join(RESULTS_DIR, f'test_{dataset_name}_{file_name}.txt')
        acc = parse_training_log(path)
        if acc:
            histories[display_name] = acc
    return histories


def main():
    datasets = [
        ('mnist',   'MNIST'),
        ('fmnist',  'Fashion-MNIST'),
        ('cifar10', 'CIFAR-10'),
    ]

    styles = {
        'SplitFed':         {'color': 'red',    'ls': '--', 'marker': 's', 'label': 'SplitFed'},
        'Multi-Tenant SFL': {'color': 'orange', 'ls': '-.', 'marker': '^', 'label': 'Multi-Tenant SFL'},
        'COIN-UCB':         {'color': 'blue',   'ls': '-',  'marker': 'D', 'label': 'LENS-UCB'},
        'Full-Info':        {'color': 'green',  'ls': '-',  'marker': 'o', 'label': 'Full-Info'},
    }

 # ── (7.0×2.2, 1×3, ) ──
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for col, (ds_key, ds_label) in enumerate(datasets):
        ax = axes[col]
        histories = load_dataset_histories(ds_key)
        for method_name, acc_list in histories.items():
            s = styles[method_name]
            rounds = list(range(1, len(acc_list) + 1))
            ax.plot(rounds, acc_list,
                    color=s['color'], linestyle=s['ls'],
                    marker=s['marker'], markersize=3, linewidth=1,
                    label=s['label'],
                    markevery=max(1, len(rounds) // 10))
        ax.set_xlabel('Training Round', fontweight='bold')
        if col == 0:
            ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(ds_label, fontsize=7, fontweight='bold')
    # Shared legend at top, horizontal
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               fontsize=6, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUTPUT_DIR, 'accuracy_convergence.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

 # ── (, ) ──
 # 7.0×2.2 → tight_layout 2.33 
    single_w = 7.0 / 3.0  # ≈ 2.33
    single_h = 2.2

    for col, (ds_key, ds_label) in enumerate(datasets):
        fig, ax = plt.subplots(figsize=(single_w, single_h))
        histories = load_dataset_histories(ds_key)
        for method_name, acc_list in histories.items():
            s = styles[method_name]
            rounds = list(range(1, len(acc_list) + 1))
            ax.plot(rounds, acc_list,
                    color=s['color'], linestyle=s['ls'],
                    marker=s['marker'], markersize=3, linewidth=1,
                    label=s['label'],
                    markevery=max(1, len(rounds) // 10))
        ax.set_xlabel('Training Round', fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=5, loc='lower right', framealpha=0.9)
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, f'accuracy_{ds_key}.pdf')
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')


if __name__ == '__main__':
    main()
