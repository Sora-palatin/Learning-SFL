"""



：
1. 1: (f_k, tau_k) → v_k ()
2. 2: (f_k, tau_k) → R_k ()
3. 3: ，(v_k, R_k) →

：
- （f，tau）（v）
- （f，tau）（v）
- ，
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

        
        Args:
 alpha:
 beta:
 mu:
 output_dir:
        """
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.output_dir = output_dir
        
 # 
        os.makedirs(output_dir, exist_ok=True)
        
 # 
        self.config = Config()
        self.config.ALPHA = alpha
        self.config.BETA = beta
        self.config.MU = mu
        
 # 
        self.physics = SystemPhysics(self.config)
        
 # f_ktau_k
        self.f_range = np.arange(0.1, 1.1, 0.1)   # computation capacity grid
        self.tau_range = np.arange(0.1, 1.1, 0.1)  # transmission delay grid
        
        print("="*80)
 print(" - ")
        print("="*80)
 print(f":")
 print(f" ALPHA = {alpha} ()")
 print(f" BETA = {beta} ()")
 print(f" MU = {mu} ()")
 print(f"\n:")
 print(f" f_k : (0.1, 1.0], =0.1")
 print(f" tau_k : (0.1, 1.0], =0.1")
 print(f" : {len(self.f_range)} × {len(self.tau_range)} = {len(self.f_range) * len(self.tau_range)} ")
 print(f"\n: {output_dir}")
        print("="*80)
    
    def compute_optimal_contract_grid(self):
        """

 (22)(23)
        
        Returns:
 v_grid: (tau × f)
 R_grid: (tau × f)
        """
 print("\n[1/4] ...")
 print("(22)(23)")
        
        n_f = len(self.f_range)
        n_tau = len(self.tau_range)
        
        v_grid = np.zeros((n_tau, n_f))
        R_grid = np.zeros((n_tau, n_f))
        
        total_points = n_f * n_tau
        computed = 0
        
 # 
        all_clients = []
        for i, tau in enumerate(self.tau_range):
            for j, f in enumerate(self.f_range):
                client = ClientType(f=f, tau=tau, data_size=50)
                all_clients.append(client)
        
 # 
        distribution_p = np.ones(total_points) / total_points
        
 # (22)(23)
        optimal_menu = self.physics.solve_optimal_contract(distribution_p, all_clients)
        
 # 
        idx = 0
        for i, tau in enumerate(self.tau_range):
            for j, f in enumerate(self.f_range):
                v_k, r_k = optimal_menu[idx]
                v_grid[i, j] = v_k
                R_grid[i, j] = r_k
                idx += 1
                
                computed += 1
                if computed % 10 == 0 or computed == total_points:
 print(f" : {computed}/{total_points} ({100*computed/total_points:.1f}%)")
        
 print(f" {total_points} ")
        
        return v_grid, R_grid
    
    def generate_heatmap1_v_distribution(self, v_grid):
        """
        Generate Heatmap 1: (f_k, tau_k) -> optimal cut layer v_k.
        x-axis: f_k (computation capacity)
        y-axis: tau_k (transmission delay)
        colour:  v_k (optimal cut layer)
        """
 print("\n[2/4] 1: (f_k vs tau_k → v_k)...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
 # 
 # RdYlBu_r(v=1,)(v=5,)
        im = ax.imshow(v_grid, cmap='RdYlBu_r', aspect='auto', origin='lower',
                      extent=[self.f_range[0]-0.05, self.f_range[-1]+0.05,
                             self.tau_range[0]-0.05, self.tau_range[-1]+0.05],
                      vmin=1, vmax=5)
        
 # 
        for i in range(len(self.f_range)+1):
            ax.axvline(self.f_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(len(self.tau_range)+1):
            ax.axhline(self.tau_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.5, alpha=0.3)
        
 # 
        ax.set_xlabel('Computing Capacity f_k', fontsize=14, fontweight='bold')
        ax.set_ylabel('Transmission Delay τ_k', fontsize=14, fontweight='bold')
        ax.set_title(f'Heatmap 1: Optimal Cut-off Layer Distribution\n(α={self.alpha}, β={self.beta}, μ={self.mu})',
                    fontsize=16, fontweight='bold')
        
 # 
        cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5])
        cbar.set_label('Optimal Cut-off Layer v*', fontsize=12, fontweight='bold')
        cbar.ax.set_yticklabels(['v=1\n(Shallow)', 'v=2', 'v=3', 'v=4', 'v=5\n(Deep)'])
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'heatmap1_v_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
 print(f" 1: {output_file}")
    
    def generate_heatmap2_R_distribution(self, R_grid):
        """
        Generate Heatmap 2: (f_k, tau_k) -> optimal reward R_k.
        x-axis: f_k (computation capacity)
        y-axis: tau_k (transmission delay)
        colour:  R_k (optimal reward)
        """
 print("\n[3/4] 2: (f_k vs tau_k → R_k)...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
 # 
        im = ax.imshow(R_grid, cmap='RdYlBu_r', aspect='auto', origin='lower',
                      extent=[self.f_range[0]-0.05, self.f_range[-1]+0.05,
                             self.tau_range[0]-0.05, self.tau_range[-1]+0.05])
        
 # 
        for i in range(len(self.f_range)+1):
            ax.axvline(self.f_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(len(self.tau_range)+1):
            ax.axhline(self.tau_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.5, alpha=0.3)
        
 # 
        ax.set_xlabel('Computing Capacity f_k', fontsize=14, fontweight='bold')
        ax.set_ylabel('Transmission Delay τ_k', fontsize=14, fontweight='bold')
        ax.set_title(f'Heatmap 2: Optimal Reward Distribution\n(α={self.alpha}, β={self.beta}, μ={self.mu})',
                    fontsize=16, fontweight='bold')
        
 # 
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Optimal Reward R*', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'heatmap2_R_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
 print(f" 2: {output_file}")
    
    def generate_heatmap3_client_utility(self, v_grid, R_grid):
        """
 3: 2-3，(v_k, R_k) →
        
 ：
 - : (f=0.3, tau=0.6) - ，
 - : (f=0.5, tau=0.5) -
 - : (f=0.6, tau=0.3) - ，
        
 (22)(23)
 ：R_k（）
 U_c = R_k - C_k
        """
 print("\n[4/4] 3: (v_k vs R_k → U_c)...")
        
 # 
 # 
        typical_clients = [
            {'name': 'Weakest Client', 'f': 0.1, 'tau': 1.0, 'color': 'blue'},
            {'name': 'Medium Client', 'f': 0.4, 'tau': 0.5, 'color': 'green'},
            {'name': 'Strongest Client', 'f': 1.0, 'tau': 0.1, 'color': 'red'}
        ]
        
 # R
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
        
 # R
        R_min = 0
        R_max = max_R_all * 1.1  # add 10% headroom for readability
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
            
 # 
            utility_grid = np.zeros((R_steps, len(v_range)))
            
            for i, R in enumerate(R_range):
                for j, v in enumerate(v_range):
 # = R - C(v)
                    C = costs[v]
                    client_utility = R - C
                    utility_grid[i, j] = client_utility
            
 # 
            im = ax.imshow(utility_grid, cmap='RdYlBu_r', aspect='auto', origin='lower',
                          extent=[0.5, 5.5, R_min, R_max])
            
 # IRR = C(v)
            IR_line = [costs[v] for v in v_range]
            ax.plot(v_range, IR_line, 'k--', linewidth=2, label='IR Constraint (R=C(v))', zorder=5)
            
 # (22)(23)
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
            
 # 
            ax.set_xlabel('Cut-off Layer v_k', fontsize=12, fontweight='bold')
            ax.set_ylabel('Reward R_k', fontsize=12, fontweight='bold')
            ax.set_title(f'{client_info["name"]}',
                        fontsize=14, fontweight='bold', color=client_info['color'])
            ax.set_xticks(v_range)
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)
            
 # 
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Client Utility U_c', fontsize=10)
        
        plt.suptitle(f'Heatmap 3: Client Utility Analysis',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'heatmap3_client_utility.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
 print(f" 3: {output_file}")
    
    def analyze_and_save_results(self, v_grid, R_grid):
        """Analyse computed grids and write a verification report to a text file."""
 print("\n[] ...")
        
        output_file = os.path.join(self.output_dir, 'theory_validation_results.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Theory Validation Results\n")
            f.write("="*80 + "\n\n")
            
            f.write("Parameter settings:\n")
            f.write(f"  ALPHA = {self.alpha} (computation cost weight)\n")
            f.write(f"  BETA = {self.beta} (transmission cost weight)\n")
            f.write(f"  MU = {self.mu} (data-volume incentive weight)\n\n")
            
            f.write("Grid settings:\n")
            f.write(f"  f_k range: (0.1, 1.0], step=0.1\n")
            f.write(f"  tau_k range: (0.1, 1.0], step=0.1\n")
            f.write(f"  Grid size: {len(self.f_range)} x {len(self.tau_range)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("Validation 1: stronger clients prefer deeper cut layers, weaker clients prefer shallower\n")
            f.write("="*80 + "\n\n")
            
 # 
 # ftau
            corners = [
                ('Weakest client  (low f, high tau)', -1, 0),  # ：tau(-1)，f(0)
                ('Strongest client (high f, low tau)', 0, -1),  # ：tau(0)，f(-1)
                ('High-compute, poor-network (high f, high tau)', -1, -1),  # ：tau(-1)，f(-1)
                ('Low-compute, good-network  (low f, low tau)',   0,  0)   # ：tau(0)，f(0)
            ]
            
            for name, i, j in corners:
                f_val = self.f_range[j]
                tau_val = self.tau_range[i]
                v_val = int(v_grid[i, j])
                R_val = R_grid[i, j]
                f.write(f"{name}:\n")
                f.write(f"  Location: f={f_val:.1f}, tau={tau_val:.1f}\n")
                f.write(f"  Optimal cut layer: v*={v_val}\n")
                f.write(f"  Optimal reward: R*={R_val:.4f}\n\n")
            
 # 
            f.write("="*80 + "\n")
            f.write("Cut-layer distribution statistics\n")
            f.write("="*80 + "\n\n")
            
            for v in range(1, 6):
                count = np.sum(v_grid == v)
                percentage = 100.0 * count / v_grid.size
                f.write(f"  v={v}: {count} grid points ({percentage:.1f}%)\n")
            
 # 
            f.write("\n" + "="*80 + "\n")
            f.write("Validation 2: one-to-one correspondence between cut layer and reward\n")
            f.write("="*80 + "\n\n")
            f.write("Explanation:\n")
            f.write("  - Optimal cut layer v* computed via Eq. (22)\n")
            f.write("  - Optimal reward R(v) computed via Eq. (23)\n")
            f.write("  - Core: reward is a function of cut layer; same v yields unique R\n")
            f.write("  - Reward baseline set by weakest client cost (tight IR constraint)\n\n")
            f.write("Cut-layer -> reward mapping:\n")
            
 # PoolingvR
            R_by_v = {}
            for v in range(1, 6):
                mask = v_grid == v
                if np.any(mask):
 # 
                    R_values = R_grid[mask]
                    unique_R = np.unique(R_values)
                    
 # PoolingvR
                    if len(unique_R) == 1:
                        R_by_v[v] = unique_R[0]
                        f.write(f"  v={v}: reward R={unique_R[0]:.4f}\n")
                    else:
 # Pooling
 # 
                        R_by_v[v] = np.max(R_values)  #
                        f.write(f"  v={v}: WARNING: {len(unique_R)} distinct reward values\n")
                        f.write(f"       range=[{np.min(R_values):.4f}, {np.max(R_values):.4f}]\n")
                        f.write(f"       using max={np.max(R_values):.4f} (ensures IR constraint)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Conclusion\n")
            f.write("="*80 + "\n\n")
            
 # 1
            weak_v = int(v_grid[-1, 0])  # ：(0.1, 1.0)
            strong_v = int(v_grid[0, -1])  # ：(1.0, 0.1)
            
            f.write("1. Cut-layer selection validation:\n")
            if strong_v > weak_v:
                f.write(f"   [PASS] Theory holds: strong client (v={strong_v}) > weak client (v={weak_v})\n")
            else:
                f.write(f"   [FAIL] Theory violated: strong client (v={strong_v}) <= weak client (v={weak_v})\n")
            
 # 2R_by_v
            f.write("\n2. Reward-cut-layer monotonicity validation:\n")
            
            monotonic = all(R_by_v[v] <= R_by_v[v+1] for v in range(1, 5) if v in R_by_v and v+1 in R_by_v)
            if monotonic:
                f.write("   [PASS] Theory holds: reward is monotonically increasing in cut layer\n")
                for v in range(1, 5):
                    if v in R_by_v and v+1 in R_by_v:
                        f.write(f"      R(v={v})={R_by_v[v]:.4f} < R(v={v+1})={R_by_v[v+1]:.4f}\n")
            else:
                f.write("   [FAIL] Theory partially violated: reward not fully monotone\n")
                for v in range(1, 5):
                    if v in R_by_v and v+1 in R_by_v:
                        if R_by_v[v] > R_by_v[v+1]:
                            f.write(f"      ✗ R(v={v})={R_by_v[v]:.4f} > R(v={v+1})={R_by_v[v+1]:.4f}\n")
                        else:
                            f.write(f"      ✓ R(v={v})={R_by_v[v]:.4f} <= R(v={v+1})={R_by_v[v+1]:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
 print(f"\n: {output_file}")
    
    def run_full_validation(self):
        """Run the full theory validation pipeline and save all outputs."""
        print("\n" + "="*80)
 print("")
        print("="*80)
        
 # 1. 
        v_grid, R_grid = self.compute_optimal_contract_grid()
        
 # 2. 
        self.generate_heatmap1_v_distribution(v_grid)
        self.generate_heatmap2_R_distribution(R_grid)
        self.generate_heatmap3_client_utility(v_grid, R_grid)
        
 # 3. 
        self.analyze_and_save_results(v_grid, R_grid)
        
        print("\n" + "="*80)
 print("")
        print("="*80)
 print(f"\n:")
        print(f"  1. {os.path.join(self.output_dir, 'heatmap1_v_distribution.png')}")
        print(f"  2. {os.path.join(self.output_dir, 'heatmap2_R_distribution.png')}")
        print(f"  3. {os.path.join(self.output_dir, 'heatmap3_client_utility.png')}")
 print(f" 4. {os.path.join(self.output_dir, '.txt')}")
        print("\n")


def main():
    """Entry point."""
    print("="*80)
 print("")
    print("="*80)
 print("\n:")
 print(" ALPHA = 0.7 ()")
 print(" BETA = 0.5 ()")
 print(" MU = 0.1 ()")
 print(" D_k = 50 ()")
 print("\n:")
 print(" - ")
 print(" - ftau")
 print(" - ")
    print("="*80)
    
 # 
    validator = TheoryValidation(
        alpha=0.7,
        beta=0.5,
        mu=0.1,
        output_dir='./TheoryValidation'
    )
    
    validator.run_full_validation()


if __name__ == '__main__':
    main()
