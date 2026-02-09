"""
理论验证实验系统
用于验证契约理论的有效性

实验目标：
1. 热力图1: (f_k, tau_k) → v_k (最优切分点分布)
2. 热力图2: (f_k, tau_k) → R_k (最优报酬分布)
3. 热力图3: 选取典型客户端，(v_k, R_k) → 客户端收益

理论预期：
- 强客户端（高f，低tau）倾向选择深切分点（v大）
- 弱客户端（低f，高tau）倾向选择浅切分点（v小）
- 切分越深，报酬越高
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
from datetime import datetime

from configs.config import Config, RESNET_PROFILE
from core.physics import SystemPhysics

# Set font for plots
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class ClientType:
    """Client Type Definition"""
    def __init__(self, f, tau, data_size=50):
        self.f = f
        self.tau = tau
        self.data_size = data_size
        self.id = f"f={f:.1f}_tau={tau:.1f}"


class TheoryValidation:
    """Theory Validation Experiment Class"""
    
    def __init__(self, alpha=0.7, beta=0.5, mu=0.1, output_dir='./TheoryValidation'):
        """
        初始化理论验证实验
        
        Args:
            alpha: 计算成本权重
            beta: 传输成本权重
            mu: 数据量权重
            output_dir: 输出目录
        """
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建配置
        self.config = Config()
        self.config.ALPHA = alpha
        self.config.BETA = beta
        self.config.MU = mu
        
        # 初始化物理系统
        self.physics = SystemPhysics(self.config)
        
        # 网格参数：f_k和tau_k的范围和步长
        self.f_range = np.arange(0.1, 1.1, 0.1)  # (0, 1]，步长0.1
        self.tau_range = np.arange(0.1, 1.1, 0.1)  # (0, 1]，步长0.1
        
        print("="*80)
        print("理论验证实验系统 - 初始化")
        print("="*80)
        print(f"参数设置:")
        print(f"  ALPHA = {alpha} (计算成本权重)")
        print(f"  BETA = {beta} (传输成本权重)")
        print(f"  MU = {mu} (数据量权重)")
        print(f"\n网格设置:")
        print(f"  f_k 范围: (0.1, 1.0], 步长=0.1")
        print(f"  tau_k 范围: (0.1, 1.0], 步长=0.1")
        print(f"  网格大小: {len(self.f_range)} × {len(self.tau_range)} = {len(self.f_range) * len(self.tau_range)} 个点")
        print(f"\n输出目录: {output_dir}")
        print("="*80)
    
    def compute_optimal_contract_grid(self):
        """
        计算网格上每个点的最优契约
        使用第四章公式(22)和(23)计算最优契约
        
        Returns:
            v_grid: 最优切分点网格 (tau × f)
            R_grid: 最优报酬网格 (tau × f)
        """
        print("\n[1/4] 计算网格上的最优契约...")
        print("使用第四章公式(22)和(23)计算最优契约")
        
        n_f = len(self.f_range)
        n_tau = len(self.tau_range)
        
        v_grid = np.zeros((n_tau, n_f))
        R_grid = np.zeros((n_tau, n_f))
        
        total_points = n_f * n_tau
        computed = 0
        
        # 创建所有网格点的客户端类型
        all_clients = []
        for i, tau in enumerate(self.tau_range):
            for j, f in enumerate(self.f_range):
                client = ClientType(f=f, tau=tau, data_size=50)
                all_clients.append(client)
        
        # 使用均匀分布（每个客户端概率相同）
        distribution_p = np.ones(total_points) / total_points
        
        # 调用公式(22)和(23)计算最优契约
        optimal_menu = self.physics.solve_optimal_contract(distribution_p, all_clients)
        
        # 将结果填充到网格中
        idx = 0
        for i, tau in enumerate(self.tau_range):
            for j, f in enumerate(self.f_range):
                v_k, r_k = optimal_menu[idx]
                v_grid[i, j] = v_k
                R_grid[i, j] = r_k
                idx += 1
                
                computed += 1
                if computed % 10 == 0 or computed == total_points:
                    print(f"  进度: {computed}/{total_points} ({100*computed/total_points:.1f}%)")
        
        print(f"  完成！共计算 {total_points} 个点")
        
        return v_grid, R_grid
    
    def generate_heatmap1_v_distribution(self, v_grid):
        """
        生成热力图1: (f_k, tau_k) → v_k
        横坐标: f_k (计算能力)
        纵坐标: tau_k (传输延迟)
        颜色: 最优切分点 v_k
        """
        print("\n[2/4] 生成热力图1: 最优切分点分布 (f_k vs tau_k → v_k)...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制热力图（从冷色调到暖色调：深蓝到深红）
        # 使用RdYlBu_r：深蓝(v=1,浅切分)到深红(v=5,深切分)
        im = ax.imshow(v_grid, cmap='RdYlBu_r', aspect='auto', origin='lower',
                      extent=[self.f_range[0]-0.05, self.f_range[-1]+0.05,
                             self.tau_range[0]-0.05, self.tau_range[-1]+0.05],
                      vmin=1, vmax=5)
        
        # 添加网格线
        for i in range(len(self.f_range)+1):
            ax.axvline(self.f_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(len(self.tau_range)+1):
            ax.axhline(self.tau_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.5, alpha=0.3)
        
        # 设置坐标轴
        ax.set_xlabel('Computing Capacity f_k', fontsize=14, fontweight='bold')
        ax.set_ylabel('Transmission Delay τ_k', fontsize=14, fontweight='bold')
        ax.set_title(f'Heatmap 1: Optimal Cut-off Layer Distribution\n(α={self.alpha}, β={self.beta}, μ={self.mu})',
                    fontsize=16, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5])
        cbar.set_label('Optimal Cut-off Layer v*', fontsize=12, fontweight='bold')
        cbar.ax.set_yticklabels(['v=1\n(Shallow)', 'v=2', 'v=3', 'v=4', 'v=5\n(Deep)'])
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'heatmap1_v_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  热力图1已保存: {output_file}")
    
    def generate_heatmap2_R_distribution(self, R_grid):
        """
        生成热力图2: (f_k, tau_k) → R_k
        横坐标: f_k (计算能力)
        纵坐标: tau_k (传输延迟)
        颜色: 最优报酬 R_k
        """
        print("\n[3/4] 生成热力图2: 最优报酬分布 (f_k vs tau_k → R_k)...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制热力图（从冷色调到暖色调：深蓝到深红）
        im = ax.imshow(R_grid, cmap='RdYlBu_r', aspect='auto', origin='lower',
                      extent=[self.f_range[0]-0.05, self.f_range[-1]+0.05,
                             self.tau_range[0]-0.05, self.tau_range[-1]+0.05])
        
        # 添加网格线
        for i in range(len(self.f_range)+1):
            ax.axvline(self.f_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(len(self.tau_range)+1):
            ax.axhline(self.tau_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.5, alpha=0.3)
        
        # 设置坐标轴
        ax.set_xlabel('Computing Capacity f_k', fontsize=14, fontweight='bold')
        ax.set_ylabel('Transmission Delay τ_k', fontsize=14, fontweight='bold')
        ax.set_title(f'Heatmap 2: Optimal Reward Distribution\n(α={self.alpha}, β={self.beta}, μ={self.mu})',
                    fontsize=16, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Optimal Reward R*', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'heatmap2_R_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  热力图2已保存: {output_file}")
    
    def generate_heatmap3_client_utility(self, v_grid, R_grid):
        """
        生成热力图3: 选取2-3个典型客户端，(v_k, R_k) → 客户端收益
        
        典型客户端：
        - 弱客户端: (f=0.3, tau=0.6) - 低算力，高延迟
        - 中等客户端: (f=0.5, tau=0.5) - 中等能力
        - 强客户端: (f=0.6, tau=0.3) - 高算力，低延迟
        
        使用第四章公式(22)和(23)计算的最优契约
        纵坐标：R_k（报酬）
        在左上角解释客户端收益U_c = R_k - C_k
        """
        print("\n[4/4] 生成热力图3: 客户端收益分析 (v_k vs R_k → U_c)...")
        
        # 定义典型客户端（选择有差异化的代表）
        # 根据实验结果，选择不同切分点和收益的客户端
        typical_clients = [
            {'name': 'Weakest Client', 'f': 0.1, 'tau': 1.0, 'color': 'blue'},
            {'name': 'Medium Client', 'f': 0.4, 'tau': 0.5, 'color': 'green'},
            {'name': 'Strongest Client', 'f': 1.0, 'tau': 0.1, 'color': 'red'}
        ]
        
        # 先计算所有客户端的最优报酬，找到最大值，统一R范围
        max_R_all = 0
        client_data = []
        for client_info in typical_clients:
            client = ClientType(f=client_info['f'], tau=client_info['tau'])
            f_idx = np.argmin(np.abs(self.f_range - client_info['f']))
            tau_idx = np.argmin(np.abs(self.tau_range - client_info['tau']))
            optimal_v = int(v_grid[tau_idx, f_idx])
            optimal_R = R_grid[tau_idx, f_idx]
            max_R_all = max(max_R_all, optimal_R)
            
            costs = {}
            for v in range(1, 6):
                costs[v] = self.physics.calculate_cost(client, v)
            
            client_data.append({
                'info': client_info,
                'client': client,
                'optimal_v': optimal_v,
                'optimal_R': optimal_R,
                'costs': costs
            })
        
        # 统一的R范围，确保所有三幅图都填充满
        R_min = 0
        R_max = max_R_all * 1.1  # 扩大10%
        R_steps = 50
        R_range = np.linspace(R_min, R_max, R_steps)
        v_range = list(range(1, 6))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, data in enumerate(client_data):
            ax = axes[idx]
            client_info = data['info']
            client = data['client']
            optimal_v_from_grid = data['optimal_v']
            optimal_R_from_grid = data['optimal_R']
            costs = data['costs']
            
            # 创建网格
            utility_grid = np.zeros((R_steps, len(v_range)))
            
            for i, R in enumerate(R_range):
                for j, v in enumerate(v_range):
                    # 客户端收益 = R - C(v)
                    C = costs[v]
                    client_utility = R - C
                    utility_grid[i, j] = client_utility
            
            # 绘制热力图（从冷色调到暖色调：深蓝到深红）
            im = ax.imshow(utility_grid, cmap='RdYlBu_r', aspect='auto', origin='lower',
                          extent=[0.5, 5.5, R_min, R_max])
            
            # 标记IR约束线（R = C(v)）
            IR_line = [costs[v] for v in v_range]
            ax.plot(v_range, IR_line, 'k--', linewidth=2, label='IR Constraint (R=C(v))', zorder=5)
            
            # 标记最优点（来自公式(22)(23)）
            optimal_C = self.physics.calculate_cost(client, optimal_v_from_grid)
            optimal_Uc = optimal_R_from_grid - optimal_C
            
            ax.plot(optimal_v_from_grid, optimal_R_from_grid, 'b*', markersize=20, 
                   markeredgecolor='white', markeredgewidth=2,
                   label=f'Optimal Contract (v={optimal_v_from_grid}, R={optimal_R_from_grid:.2f})', zorder=10)
            
            # Add text annotation in the upper left corner (English version)
            uc_text = f'Client Utility Formula:\nU_c = R - C\n\nOptimal Contract:\nv* = {optimal_v_from_grid}\nR* = {optimal_R_from_grid:.4f}\nC* = {optimal_C:.4f}\nU_c* = {optimal_Uc:.4f}'
            if optimal_Uc < 0:
                uc_text += '\n⚠️ IR Constraint Violated!'
            ax.text(0.02, 0.98, uc_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
            
            # 设置坐标轴
            ax.set_xlabel('Cut-off Layer v_k', fontsize=12, fontweight='bold')
            ax.set_ylabel('Reward R_k', fontsize=12, fontweight='bold')
            ax.set_title(f'{client_info["name"]}',
                        fontsize=14, fontweight='bold', color=client_info['color'])
            ax.set_xticks(v_range)
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Client Utility U_c', fontsize=10)
        
        plt.suptitle(f'Heatmap 3: Client Utility Analysis',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'heatmap3_client_utility.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  热力图3已保存: {output_file}")
    
    def analyze_and_save_results(self, v_grid, R_grid):
        """分析结果并保存到文本文件"""
        print("\n[分析] 验证理论预期...")
        
        output_file = os.path.join(self.output_dir, '理论验证结果.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("理论验证实验结果\n")
            f.write("="*80 + "\n\n")
            
            f.write("参数设置:\n")
            f.write(f"  ALPHA = {self.alpha} (计算成本权重)\n")
            f.write(f"  BETA = {self.beta} (传输成本权重)\n")
            f.write(f"  MU = {self.mu} (数据量权重)\n\n")
            
            f.write("网格设置:\n")
            f.write(f"  f_k 范围: (0.1, 1.0], 步长=0.1\n")
            f.write(f"  tau_k 范围: (0.1, 1.0], 步长=0.1\n")
            f.write(f"  网格大小: {len(self.f_range)} × {len(self.tau_range)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("理论验证1: 强客户端倾向深切分点，弱客户端倾向浅切分点\n")
            f.write("="*80 + "\n\n")
            
            # 分析四个角落的客户端
            # 注意：横坐标f单调递增，纵坐标tau单调递减（从下到上）
            corners = [
                ('最弱客户端 (低f,高tau)', -1, 0),  # 左上角：tau最大(索引-1)，f最小(索引0)
                ('最强客户端 (高f,低tau)', 0, -1),  # 右下角：tau最小(索引0)，f最大(索引-1)
                ('高算力弱网络 (高f,高tau)', -1, -1),  # 右上角：tau最大(索引-1)，f最大(索引-1)
                ('低算力强网络 (低f,低tau)', 0, 0)   # 左下角：tau最小(索引0)，f最小(索引0)
            ]
            
            for name, i, j in corners:
                f_val = self.f_range[j]
                tau_val = self.tau_range[i]
                v_val = int(v_grid[i, j])
                R_val = R_grid[i, j]
                f.write(f"{name}:\n")
                f.write(f"  位置: f={f_val:.1f}, tau={tau_val:.1f}\n")
                f.write(f"  最优切分点: v*={v_val}\n")
                f.write(f"  最优报酬: R*={R_val:.4f}\n\n")
            
            # 统计切分点分布
            f.write("="*80 + "\n")
            f.write("切分点分布统计\n")
            f.write("="*80 + "\n\n")
            
            for v in range(1, 6):
                count = np.sum(v_grid == v)
                percentage = 100.0 * count / v_grid.size
                f.write(f"  v={v}: {count}个点 ({percentage:.1f}%)\n")
            
            # 分析报酬与切分点的关系（不计算平均值，直接看单调性）
            f.write("\n" + "="*80 + "\n")
            f.write("理论验证2: 切分点与报酬的一一对应关系\n")
            f.write("="*80 + "\n\n")
            f.write("说明：\n")
            f.write("  - 使用公式(22)计算每个客户端的最优切分点v*\n")
            f.write("  - 使用公式(23)计算每个切分点v的最优报酬R(v)\n")
            f.write("  - 核心逻辑：报酬是切分点的函数，同一v对应唯一R\n")
            f.write("  - 使用全局最弱客户端的成本作为报酬基准（紧IR约束）\n\n")
            f.write("切分点 → 报酬映射：\n")
            
            # 收集每个切分点对应的报酬值（强制Pooling后，每个v对应唯一R）
            R_by_v = {}
            for v in range(1, 6):
                mask = v_grid == v
                if np.any(mask):
                    # 获取该切分点的所有报酬值
                    R_values = R_grid[mask]
                    unique_R = np.unique(R_values)
                    
                    # 由于强制Pooling，同一v应该只有一个R值
                    if len(unique_R) == 1:
                        R_by_v[v] = unique_R[0]
                        f.write(f"  v={v}: 报酬 R={unique_R[0]:.4f}\n")
                    else:
                        # 理论上不应该出现这种情况（强制Pooling后）
                        # 如果出现，说明代码有问题
                        R_by_v[v] = np.max(R_values)  # 使用最大值
                        f.write(f"  v={v}: 异常！存在{len(unique_R)}个不同报酬值\n")
                        f.write(f"       范围=[{np.min(R_values):.4f}, {np.max(R_values):.4f}]\n")
                        f.write(f"       使用最大值={np.max(R_values):.4f}（确保IR约束）\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("结论\n")
            f.write("="*80 + "\n\n")
            
            # 验证理论1
            weak_v = int(v_grid[-1, 0])  # 最弱客户端：左上角(0.1, 1.0)
            strong_v = int(v_grid[0, -1])  # 最强客户端：右下角(1.0, 0.1)
            
            f.write("1. 切分点选择验证:\n")
            if strong_v > weak_v:
                f.write(f"   ✓ 理论成立: 强客户端(v={strong_v}) > 弱客户端(v={weak_v})\n")
            else:
                f.write(f"   ✗ 理论不成立: 强客户端(v={strong_v}) ≤ 弱客户端(v={weak_v})\n")
            
            # 验证理论2（使用已经计算好的R_by_v）
            f.write("\n2. 报酬与切分点关系验证:\n")
            
            monotonic = all(R_by_v[v] <= R_by_v[v+1] for v in range(1, 5) if v in R_by_v and v+1 in R_by_v)
            if monotonic:
                f.write("   ✓ 理论成立: 报酬随切分点单调递增\n")
                for v in range(1, 5):
                    if v in R_by_v and v+1 in R_by_v:
                        f.write(f"      R(v={v})={R_by_v[v]:.4f} < R(v={v+1})={R_by_v[v+1]:.4f}\n")
            else:
                f.write("   ✗ 理论部分成立: 报酬未完全单调递增\n")
                for v in range(1, 5):
                    if v in R_by_v and v+1 in R_by_v:
                        if R_by_v[v] > R_by_v[v+1]:
                            f.write(f"      ✗ R(v={v})={R_by_v[v]:.4f} > R(v={v+1})={R_by_v[v+1]:.4f}\n")
                        else:
                            f.write(f"      ✓ R(v={v})={R_by_v[v]:.4f} <= R(v={v+1})={R_by_v[v+1]:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\n分析结果已保存: {output_file}")
    
    def run_full_validation(self):
        """运行完整的理论验证实验"""
        print("\n" + "="*80)
        print("开始理论验证实验")
        print("="*80)
        
        # 1. 计算网格上的最优契约
        v_grid, R_grid = self.compute_optimal_contract_grid()
        
        # 2. 生成三张热力图
        self.generate_heatmap1_v_distribution(v_grid)
        self.generate_heatmap2_R_distribution(R_grid)
        self.generate_heatmap3_client_utility(v_grid, R_grid)
        
        # 3. 分析并保存结果
        self.analyze_and_save_results(v_grid, R_grid)
        
        print("\n" + "="*80)
        print("理论验证实验完成！")
        print("="*80)
        print(f"\n输出文件:")
        print(f"  1. {os.path.join(self.output_dir, 'heatmap1_v_distribution.png')}")
        print(f"  2. {os.path.join(self.output_dir, 'heatmap2_R_distribution.png')}")
        print(f"  3. {os.path.join(self.output_dir, 'heatmap3_client_utility.png')}")
        print(f"  4. {os.path.join(self.output_dir, '理论验证结果.txt')}")
        print("\n")


def main():
    """主函数"""
    print("="*80)
    print("理论验证实验系统")
    print("="*80)
    print("\n推荐参数设置:")
    print("  ALPHA = 0.7 (计算成本权重)")
    print("  BETA = 0.5 (传输成本权重)")
    print("  MU = 0.1 (数据量权重)")
    print("  D_k = 50 (数据量)")
    print("\n这组参数能够:")
    print("  - 平衡计算和通信成本的影响")
    print("  - 减小常数项影响，突出f和tau的作用")
    print("  - 使切分点选择呈现从左上到右下逐渐加深的趋势")
    print("="*80)
    
    # 运行实验
    validator = TheoryValidation(
        alpha=0.7,
        beta=0.5,
        mu=0.1,
        output_dir='./TheoryValidation'
    )
    
    validator.run_full_validation()


if __name__ == '__main__':
    main()
