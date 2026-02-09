"""
KDD论文热力图生成脚本
- 热力图1/2: figsize=(3.5, 3.0), 单栏宽度, fontsize=10, 无标题
- 热力图3: figsize=(7.0, 2.5), 双栏宽度, 1×3子图, aspect='auto', 无标题
- 输出PDF到 new_figure/ 目录
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from configs.config import Config, RESNET_PROFILE
from core.physics import SystemPhysics

# ── 全局字体设置 ──
rcParams.update({
    'font.size': 18,
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,      # TrueType fonts in PDF (editable text)
    'ps.fonttype': 42,
})

OUTPUT_DIR = os.path.dirname(__file__)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 数据计算（复用 theory_validation 逻辑）──

class ClientType:
    def __init__(self, f, tau, data_size=50):
        self.f = f
        self.tau = tau
        self.data_size = data_size
        self.id = f"f={f:.1f}_tau={tau:.1f}"


def compute_grids(alpha=0.7, beta=0.5, mu=0.1):
    config = Config()
    config.ALPHA = alpha
    config.BETA = beta
    config.MU = mu
    physics = SystemPhysics(config)

    f_range = np.arange(0.1, 1.1, 0.1)
    tau_range = np.arange(0.1, 1.1, 0.1)

    n_f, n_tau = len(f_range), len(tau_range)
    total = n_f * n_tau

    all_clients = []
    for tau in tau_range:
        for f in f_range:
            all_clients.append(ClientType(f=f, tau=tau))

    distribution_p = np.ones(total) / total
    optimal_menu = physics.solve_optimal_contract(distribution_p, all_clients)

    v_grid = np.zeros((n_tau, n_f))
    R_grid = np.zeros((n_tau, n_f))
    idx = 0
    for i in range(n_tau):
        for j in range(n_f):
            v_grid[i, j], R_grid[i, j] = optimal_menu[idx]
            idx += 1

    return v_grid, R_grid, f_range, tau_range, physics


# ── 热力图 1+2 合并: V分布 + R分布 (单画布, 严格对齐) ──

def plot_heatmap12_combined(v_grid, R_grid, f_range, tau_range):
    """在一张画布上绘制 V 图(左) 和 R 图(右), 使用 make_axes_locatable
    固定 colorbar 宽度, 确保两个子图主体区域严格一致."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ext = [f_range[0]-0.05, f_range[-1]+0.05,
           tau_range[0]-0.05, tau_range[-1]+0.05]

    # ── 左: V 图 (Optimal Cut-off Layer) ──
    im1 = ax1.imshow(v_grid, cmap='RdYlBu_r', aspect='auto', origin='lower',
                     extent=ext, vmin=1, vmax=5)
    for i in range(len(f_range)+1):
        ax1.axvline(f_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.3, alpha=0.3)
    for i in range(len(tau_range)+1):
        ax1.axhline(tau_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.3, alpha=0.3)
    # 刻度放在网格线边缘(右/上边界): 0.2→0.25, 0.4→0.45, 0.6→0.65, 0.8→0.85, 1.0→1.05
    tick_positions = [0.25, 0.45, 0.65, 0.85, 1.05]
    tick_labels = ['0.2', '0.4', '0.6', '0.8', '1.0']
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels)
    ax1.set_xlabel('Computing Capacity $f_k$', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Transmission Delay $\\tau_k$', fontsize=11, fontweight='bold')

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax1, ticks=[1, 2, 3, 4, 5])
    cbar1.set_label('Optimal Cut-off Layer $v^*$', fontsize=10, fontweight='bold')
    cbar1.ax.set_yticklabels(['$v$=1\n(Shallow)', '$v$=2', '$v$=3', '$v$=4', '$v$=5\n(Deep)'],
                             fontsize=8)

    # ── 右: R 图 (Optimal Reward) ──
    im2 = ax2.imshow(R_grid, cmap='RdYlBu_r', aspect='auto', origin='lower',
                     extent=ext)
    for i in range(len(f_range)+1):
        ax2.axvline(f_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.3, alpha=0.3)
    for i in range(len(tau_range)+1):
        ax2.axhline(tau_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.3, alpha=0.3)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels)
    ax2.set_xlabel('Computing Capacity $f_k$', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Transmission Delay $\\tau_k$', fontsize=11, fontweight='bold')

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('Optimal Reward $R^*$', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'heatmap_vR.pdf')
    plt.savefig(out, bbox_inches='tight')
    print(f'  Saved: {out}')

    # ── 拆分保存: 从同一画布提取各子图, 保证尺寸严格一致 ──
    for ax_i, cax_i, fname in [(ax1, cax1, 'heatmap1.pdf'),
                                (ax2, cax2, 'heatmap2.pdf')]:
        # 合并 ax 和 colorbar 的 bounding box
        renderer = fig.canvas.get_renderer()
        bb_ax = ax_i.get_tightbbox(renderer)
        bb_cb = cax_i.get_tightbbox(renderer)
        bb = bb_ax.union([bb_ax, bb_cb])
        bb_expanded = bb.expanded(1.08, 1.08)  # 留少量边距
        out_i = os.path.join(OUTPUT_DIR, fname)
        fig.savefig(out_i, bbox_inches=bb_expanded.transformed(fig.dpi_scale_trans.inverted()))
        print(f'  Saved: {out_i}')

    plt.close()


# ── 热力图 3: 客户端收益 (1×3 子图) ──

def plot_heatmap3(v_grid, R_grid, f_range, tau_range, physics):
    typical_clients = [
        {'name': 'Weakest Client', 'f': 0.1, 'tau': 1.0, 'color': 'blue'},
        {'name': 'Medium Client',  'f': 0.4, 'tau': 0.5, 'color': 'green'},
        {'name': 'Strongest Client', 'f': 1.0, 'tau': 0.1, 'color': 'red'},
    ]

    # 预计算
    max_R_all = 0
    vmin_all, vmax_all = np.inf, -np.inf
    client_data = []
    for ci in typical_clients:
        client = ClientType(f=ci['f'], tau=ci['tau'])
        f_idx = np.argmin(np.abs(f_range - ci['f']))
        tau_idx = np.argmin(np.abs(tau_range - ci['tau']))
        optimal_v = int(v_grid[tau_idx, f_idx])
        optimal_R = R_grid[tau_idx, f_idx]
        max_R_all = max(max_R_all, optimal_R)
        costs = {v: physics.calculate_cost(client, v) for v in range(1, 6)}
        client_data.append({
            'info': ci, 'client': client,
            'optimal_v': optimal_v, 'optimal_R': optimal_R, 'costs': costs,
        })

    R_min, R_max = 0, max_R_all * 1.1
    R_steps = 50
    R_range = np.linspace(R_min, R_max, R_steps)
    v_range = list(range(1, 6))

    # 预计算所有utility范围以统一colorbar
    for data in client_data:
        costs = data['costs']
        for R in R_range:
            for v in v_range:
                u = R - costs[v]
                vmin_all = min(vmin_all, u)
                vmax_all = max(vmax_all, u)

    # figsize=(10, 3), 子图间距极小, 右侧留空给共享colorbar
    fig, axes = plt.subplots(1, 3, figsize=(7, 3))
    fig.subplots_adjust(left=0.06, right=0.88, wspace=0.05)
    plt.rcParams.update({'font.size': 18})

    ims = []
    for idx, data in enumerate(client_data):
        ax = axes[idx]
        ci = data['info']
        client = data['client']
        optimal_v = data['optimal_v']
        optimal_R = data['optimal_R']
        costs = data['costs']

        utility_grid = np.zeros((R_steps, len(v_range)))
        for i, R in enumerate(R_range):
            for j, v in enumerate(v_range):
                utility_grid[i, j] = R - costs[v]

        im = ax.imshow(utility_grid, cmap='RdYlBu_r', aspect='auto', origin='lower',
                       extent=[0.5, 5.5, R_min, R_max],
                       vmin=vmin_all, vmax=vmax_all)
        ims.append(im)

        # IR constraint line
        IR_line = [costs[v] for v in v_range]
        ax.plot(v_range, IR_line, 'k--', linewidth=1.2, label='IR: $R=C(v)$', zorder=5)

        # Optimal point
        optimal_C = physics.calculate_cost(client, optimal_v)
        optimal_Uc = optimal_R - optimal_C
        ax.plot(optimal_v, optimal_R, 'b*', markersize=10,
                markeredgecolor='white', markeredgewidth=1,
                label=(f'$v^*={optimal_v},\\ R^*={optimal_R:.2f}$\n'
                       f'$U_c^*={optimal_Uc:.4f}$'),
                zorder=10)

        # x轴标签只在中间子图显示
        ax.set_xticks(v_range)
        if idx == 1:
            ax.set_xlabel('Cut-off Layer $v_k$', fontweight='bold')
        else:
            ax.set_xlabel('')

        # y轴标签只在最左子图
        if idx == 0:
            ax.set_ylabel('Reward $R_k$', fontweight='bold')
        else:
            ax.set_yticklabels([])

        leg = ax.legend(fontsize=8, loc='upper left',
                        borderaxespad=0.3, handletextpad=0.4)
        ax.grid(True, alpha=0.3)

        # 子图标题
        ax.set_title(ci['name'], fontsize=10, fontweight='bold', color=ci['color'])

    # 共享 Colorbar: 放在最右侧, 高度与子图一致
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.72])  # [left, bottom, width, height]
    cbar = fig.colorbar(ims[-1], cax=cbar_ax, ticks=[-100, -50, 0, 50, 100])
    cbar.set_label('$U_c$', fontweight='bold')

    out_pdf = os.path.join(OUTPUT_DIR, 'heatmap3.pdf')
    out_png = os.path.join(OUTPUT_DIR, 'heatmap3.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    print(f'  Saved: {out_png}')
    try:
        fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
        print(f'  Saved: {out_pdf}')
    except PermissionError:
        print(f'  WARNING: {out_pdf} is locked, skipping PDF save.')
    plt.close()


# ── Main ──

if __name__ == '__main__':
    print('Computing optimal contract grids...')
    v_grid, R_grid, f_range, tau_range, physics = compute_grids()
    print('Generating KDD-formatted heatmaps...')
    plot_heatmap12_combined(v_grid, R_grid, f_range, tau_range)
    plot_heatmap3(v_grid, R_grid, f_range, tau_range, physics)
    print('Done. All PDFs saved to new_figure/')
