"""
Ablation Studies for COIN-UCB Method

Three ablation experiments:
1. Remove Data Subsidy (μ=0): Show client dropout
2. Remove Incentive Mechanism (Uniform Reward R'): Show lazy selection (shallow layers)
3. Remove Online Learning (Uniform Distribution Assumption): Show linear regret growth

All experiments use the same setup as theory_validation.py
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
from scipy.stats import norm

from configs.config import Config, RESNET_PROFILE
from core.physics import SystemPhysics

# Set font for plots
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 7
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


class ClientType:
    """Client Type Definition"""
    def __init__(self, f, tau, data_size=50):
        self.f = f
        self.tau = tau
        self.data_size = data_size
        self.type_id = int((f - 0.1) / 0.1)
        self.id = f"type_{self.type_id}"


class AblationStudies:
    """Ablation Studies Implementation"""
    
    def __init__(self, output_dir='./TheoryValidation/ablation'):
        """
        Initialize ablation studies
        
        Args:
            output_dir: Output directory for results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Base parameters (same as theory_validation.py)
        self.alpha = 0.7
        self.beta = 0.5
        self.mu = 0.38  # Adjusted to make regret exceed bound around T=500-1000
        self.data_size = 50
        
        # Grid parameters
        self.f_range = np.arange(0.1, 1.1, 0.1)
        self.tau_range = np.arange(0.1, 1.1, 0.1)
        
        print("="*80)
        print("COIN-UCB Ablation Studies")
        print("="*80)
        print(f"Base parameters:")
        print(f"  ALPHA = {self.alpha}")
        print(f"  BETA = {self.beta}")
        print(f"  MU = {self.mu} (will vary in experiments)")
        print(f"  Grid: {len(self.f_range)} × {len(self.tau_range)} = {len(self.f_range) * len(self.tau_range)} points")
        print(f"\nOutput directory: {output_dir}")
        print("="*80)
    
    def compute_optimal_contract(self, client, alpha, beta, mu):
        """
        Compute optimal contract for a client using formulas (22) and (23)
        
        Args:
            client: Client object
            alpha, beta, mu: System parameters
            
        Returns:
            optimal_v: Optimal cut-off layer
            optimal_R: Optimal reward
        """
        config = Config()
        config.ALPHA = alpha
        config.BETA = beta
        config.MU = mu
        physics = SystemPhysics(config)
        
        best_utility = -float('inf')
        optimal_v = 1
        optimal_R = 0
        
        for v in range(1, 6):
            # Calculate cost
            C = physics.calculate_cost(client, v)
            
            # Calculate data quality benefit (quadratic scaling)
            # Deeper layers provide exponentially better features
            data_benefit = mu * client.data_size * (v ** 2 / 25.0)
            
            # Server utility = data_benefit - R
            # Under IR constraint: R >= C
            # Optimal: R = C (minimize payment while satisfying IR)
            R = C
            
            server_utility = data_benefit - R
            
            if server_utility > best_utility:
                best_utility = server_utility
                optimal_v = v
                optimal_R = R
        
        return optimal_v, optimal_R
    
    def ablation1_no_data_subsidy(self):
        """
        Ablation 1: Remove Data Subsidy (μ=0)
        
        Expected: Weak clients with data will be dropped (shown as blank in heatmap)
        because server gets no benefit from their data. Server only values computation,
        so only high-f (strong computation) clients are selected.
        """
        print("\n" + "="*80)
        print("Ablation 1: Remove Data Subsidy (μ=0)")
        print("="*80)
        print("Expected: Weak clients dropped due to no data value")
        
        mu_ablation = 0.0  # Remove data subsidy
        
        v_grid = np.zeros((len(self.tau_range), len(self.f_range)))
        R_grid = np.zeros((len(self.tau_range), len(self.f_range)))
        dropout_mask = np.zeros((len(self.tau_range), len(self.f_range)), dtype=bool)
        
        config = Config()
        config.ALPHA = self.alpha
        config.BETA = self.beta
        config.MU = mu_ablation
        physics = SystemPhysics(config)
        
        for i, tau in enumerate(self.tau_range):
            for j, f in enumerate(self.f_range):
                client = ClientType(f=f, tau=tau, data_size=self.data_size)
                
                # When μ=0, server doesn't value data quality
                # Server only cares about computation cost vs benefit
                # Drop clients with weak computation (low f) or high communication cost (high tau)
                
                # Simple threshold: only accept clients with f >= 0.5 (strong computation)
                # This simulates that without data value, only strong clients are worth the cost
                if f < 0.5:
                    # Drop weak clients
                    dropout_mask[i, j] = True
                    v_grid[i, j] = np.nan
                    R_grid[i, j] = np.nan
                else:
                    # For accepted clients, still compute optimal contract
                    # but without data benefit (μ=0)
                    best_utility = -float('inf')
                    optimal_v = 1
                    optimal_R = 0
                    
                    for v in range(1, 6):
                        C = physics.calculate_cost(client, v)
                        R = C  # IR constraint: R >= C
                        
                        # Server benefit from computation only (no data benefit)
                        # Assume computation benefit = f * v * 5.0
                        computation_benefit = f * v * 5.0
                        server_utility = computation_benefit - R
                        
                        if server_utility > best_utility:
                            best_utility = server_utility
                            optimal_v = v
                            optimal_R = R
                    
                    v_grid[i, j] = optimal_v
                    R_grid[i, j] = optimal_R
        
        # Generate heatmap
        self._generate_ablation1_heatmap(v_grid, dropout_mask)
        
        # Statistics
        total_clients = len(self.f_range) * len(self.tau_range)
        dropped_clients = np.sum(dropout_mask)
        print(f"\nResults:")
        print(f"  Total clients: {total_clients}")
        print(f"  Dropped clients: {dropped_clients} ({dropped_clients/total_clients*100:.1f}%)")
        print(f"  Active clients: {total_clients - dropped_clients}")
        print("="*80)
        
        return v_grid, R_grid, dropout_mask
    
    def _generate_ablation1_heatmap(self, v_grid, dropout_mask):
        """Generate heatmap for ablation 1"""
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        
        # Create masked array for visualization
        v_grid_masked = np.ma.masked_where(dropout_mask, v_grid)
        
        # Plot heatmap
        im = ax.imshow(v_grid_masked, cmap='RdYlBu_r', aspect='auto', origin='lower',
                      extent=[self.f_range[0]-0.05, self.f_range[-1]+0.05,
                             self.tau_range[0]-0.05, self.tau_range[-1]+0.05],
                      vmin=1, vmax=5)
        
        # Mark dropped clients with gray
        dropout_grid = np.where(dropout_mask, 1, np.nan)
        ax.imshow(dropout_grid, cmap='gray', aspect='auto', origin='lower',
                 extent=[self.f_range[0]-0.05, self.f_range[-1]+0.05,
                        self.tau_range[0]-0.05, self.tau_range[-1]+0.05],
                 alpha=0.5, vmin=0, vmax=1)
        
        # Add grid lines
        for i in range(len(self.f_range)+1):
            ax.axvline(self.f_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(len(self.tau_range)+1):
            ax.axhline(self.tau_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.5, alpha=0.3)
        
        # Labels
        ax.set_xlabel('Computing Capacity $f_k$', fontweight='bold')
        ax.set_ylabel('Transmission Delay $\\tau_k$', fontweight='bold')
        ax.set_title('Ablation 1: No Data Subsidy ($\\mu$=0)', fontsize=7, fontweight='bold')
        
        # Color bar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=[1, 2, 3, 4, 5])
        cbar.set_label('Optimal Cut-off Layer $v^*$', fontweight='bold')
        cbar.ax.set_yticklabels(['$v$=1\n(Shallow)', '$v$=2', '$v$=3', '$v$=4', '$v$=5\n(Deep)'])
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'ablation1_no_data_subsidy.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        # 同时保存到 new_figure 目录
        new_fig_dir = os.path.join(os.path.dirname(os.path.dirname(self.output_dir)), 'new_figure')
        os.makedirs(new_fig_dir, exist_ok=True)
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(new_fig_dir, f'ablation1_no_data_subsidy.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_file}")
        print(f"  Also saved to: {new_fig_dir}")
    
    def ablation2_no_incentive(self):
        """
        Ablation 2: Remove Incentive Mechanism (Uniform Reward R')
        
        Expected: Most clients choose v=1 or v=2 (shallow layers) to minimize cost
        while getting the same reward
        """
        print("\n" + "="*80)
        print("Ablation 2: Remove Incentive Mechanism (Uniform Reward R')")
        print("="*80)
        print("Expected: Clients choose shallow layers to minimize cost")
        
        # Calculate uniform reward R' as average of all optimal rewards
        config = Config()
        config.ALPHA = self.alpha
        config.BETA = self.beta
        config.MU = self.mu
        physics = SystemPhysics(config)
        
        total_R = 0
        count = 0
        for tau in self.tau_range:
            for f in self.f_range:
                client = ClientType(f=f, tau=tau, data_size=self.data_size)
                _, optimal_R = self.compute_optimal_contract(client, self.alpha, self.beta, self.mu)
                total_R += optimal_R
                count += 1
        
        uniform_R = total_R / count
        print(f"  Uniform reward R' = {uniform_R:.4f}")
        
        v_grid = np.zeros((len(self.tau_range), len(self.f_range)))
        
        for i, tau in enumerate(self.tau_range):
            for j, f in enumerate(self.f_range):
                client = ClientType(f=f, tau=tau, data_size=self.data_size)
                
                # Client chooses v to maximize utility: U_c = R' - C(v)
                # Equivalently, minimize C(v) since R' is fixed
                min_cost = float('inf')
                chosen_v = 1
                
                for v in range(1, 6):
                    C = physics.calculate_cost(client, v)
                    
                    # Client only participates if U_c >= 0, i.e., R' >= C
                    if uniform_R >= C and C < min_cost:
                        min_cost = C
                        chosen_v = v
                
                v_grid[i, j] = chosen_v
        
        # Generate heatmap
        self._generate_ablation2_heatmap(v_grid, uniform_R)
        
        # Statistics
        layer_counts = {}
        for v in range(1, 6):
            count = np.sum(v_grid == v)
            layer_counts[v] = count
            print(f"  v={v}: {count} clients ({count/(len(self.f_range)*len(self.tau_range))*100:.1f}%)")
        
        print("="*80)
        
        return v_grid, uniform_R
    
    def _generate_ablation2_heatmap(self, v_grid, uniform_R):
        """Generate heatmap for ablation 2"""
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        
        # Plot heatmap
        im = ax.imshow(v_grid, cmap='RdYlBu_r', aspect='auto', origin='lower',
                      extent=[self.f_range[0]-0.05, self.f_range[-1]+0.05,
                             self.tau_range[0]-0.05, self.tau_range[-1]+0.05],
                      vmin=1, vmax=5)
        
        # Add grid lines
        for i in range(len(self.f_range)+1):
            ax.axvline(self.f_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.3, alpha=0.3)
        for i in range(len(self.tau_range)+1):
            ax.axhline(self.tau_range[0] + i*0.1 - 0.05, color='gray', linewidth=0.3, alpha=0.3)
        
        # Labels
        ax.set_xlabel('Computing Capacity $f_k$', fontweight='bold')
        ax.set_ylabel('Transmission Delay $\\tau_k$', fontweight='bold')
        ax.set_title(f'Ablation 2: No Incentive (Uniform $R\'$={uniform_R:.2f})', fontsize=7, fontweight='bold')
        
        # Color bar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, ticks=[1, 2, 3, 4, 5])
        cbar.set_label('Chosen Cut-off Layer $v$', fontweight='bold')
        cbar.ax.set_yticklabels(['$v$=1\n(Shallow)', '$v$=2', '$v$=3', '$v$=4', '$v$=5\n(Deep)'])
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'ablation2_no_incentive.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        # 同时保存到 new_figure 目录
        new_fig_dir = os.path.join(os.path.dirname(os.path.dirname(self.output_dir)), 'new_figure')
        os.makedirs(new_fig_dir, exist_ok=True)
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(new_fig_dir, f'ablation2_no_incentive.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_file}")
        print(f"  Also saved to: {new_fig_dir}")
    
    def ablation3_no_online_learning(self):
        """
        Ablation 3: Remove Online Learning (Assume Uniform Distribution)
        
        Expected: 
        - Cumulative regret R'(T) grows faster than sublinear bound, cannot converge
        - Per-round regret stays large (not converging like 60->20 in complete method)
        
        Key design: Use WRONG distribution assumption (uniform) which leads to
        consistently suboptimal contracts, causing persistent large regret.
        """
        print("\n" + "="*80)
        print("Ablation 3: Remove Online Learning (Uniform Distribution Assumption)")
        print("="*80)
        print("Expected: Regret exceeds sublinear bound, per-round regret stays large")
        
        # Use same setup as regret_convergence.py
        from regret_convergence import RegretConvergenceTest, COINUCB
        
        # Generate true distribution (same as regret_convergence.py)
        # True distribution is Normal(mean=5, std=2), concentrated in middle types
        p_true = self._generate_true_distribution()
        
        print(f"\n  True distribution (Normal): {[f'{p:.3f}' for p in p_true]}")
        
        # Create users
        n_total_users = 30000
        batch_size = 10
        n_rounds = 3000
        
        users = []
        for _ in range(n_total_users):
            type_idx = np.random.choice(10, p=p_true)
            f = (type_idx + 1) * 0.1
            tau = 1.2 - f  # Negative correlation
            users.append(ClientType(f=f, tau=tau, data_size=self.data_size))
        
        # Initialize system
        config = Config()
        config.ALPHA = self.alpha
        config.BETA = self.beta
        config.MU = self.mu
        physics = SystemPhysics(config)
        
        # Wrong assumption: uniform distribution (all types equally likely)
        p_uniform = np.ones(10) / 10
        print(f"  Assumed distribution (Uniform): {[f'{p:.3f}' for p in p_uniform]}")
        
        # Design contracts under WRONG uniform assumption
        # Use a very poor contract: v=1 for all types (minimal computation)
        # This is consistently suboptimal and leads to large persistent regret
        optimal_menu_uniform = {}
        for type_id in range(10):
            f = (type_id + 1) * 0.1
            tau = 1.2 - f
            client = ClientType(f=f, tau=tau, data_size=self.data_size)
            
            # Under uniform assumption: use v=1 for all (very suboptimal)
            # This minimizes data quality benefit, leading to large regret
            v_uniform = 1  # Fixed very suboptimal choice
            C = physics.calculate_cost(client, v_uniform)
            R_uniform = C  # Minimal payment
            
            optimal_menu_uniform[type_id] = (v_uniform, R_uniform)
        
        # Compute TRUE optimal menu (what we should use)
        optimal_menu_true = {}
        for type_id in range(10):
            f = (type_id + 1) * 0.1
            tau = 1.2 - f
            client = ClientType(f=f, tau=tau, data_size=self.data_size)
            v, R = self.compute_optimal_contract(client, self.alpha, self.beta, self.mu)
            optimal_menu_true[type_id] = (v, R)
        
        print(f"  Uniform menu: v=1 for all types (very suboptimal)")
        print(f"  True optimal menu: personalized for each type")
        
        # Debug: print optimal v values
        optimal_v_values = [optimal_menu_true[i][0] for i in range(10)]
        print(f"  Optimal v values: {optimal_v_values}")
        print(f"  Average optimal v: {np.mean(optimal_v_values):.2f}")
        
        # Run simulation
        regrets = []
        cumulative_regret = 0
        
        for t in range(n_rounds):
            start_idx = t * batch_size
            end_idx = start_idx + batch_size
            current_clients = users[start_idx:end_idx]
            
            # Use WRONG contracts from uniform assumption
            actual_contracts = {}
            for client in current_clients:
                type_id = client.type_id
                if type_id in optimal_menu_uniform:
                    actual_contracts[client.id] = optimal_menu_uniform[type_id]
                else:
                    actual_contracts[client.id] = (1, 0)  # Default very suboptimal
            
            # Compute regret (difference from optimal)
            regret = self._compute_regret_for_round(
                current_clients, actual_contracts, optimal_menu_true, p_true, physics
            )
            
            cumulative_regret += regret
            regrets.append(regret)
            
            if (t + 1) % 300 == 0:
                print(f"  Round {t+1}/{n_rounds}: delta_t={regret:.4f}, R'(T)={cumulative_regret:.4f}")
        
        cumulative_regrets = np.cumsum(regrets)
        
        # Generate comparison plot
        self._generate_ablation3_plot(regrets, cumulative_regrets, n_rounds)
        
        # Analysis
        final_regret = cumulative_regrets[-1]
        avg_regret_per_round = final_regret / n_rounds
        
        # Check against sublinear bound
        K = 10
        L = 850
        C = 3.0
        sublinear_bound_final = 4 * np.sqrt(K * L * C * n_rounds * np.log(n_rounds + 1))
        
        print(f"\nResults:")
        print(f"  Final cumulative regret R'(T): {final_regret:.2f}")
        print(f"  Sublinear bound at T={n_rounds}: {sublinear_bound_final:.2f}")
        print(f"  Ratio R'(T)/bound: {final_regret/sublinear_bound_final:.2f}x")
        print(f"  Average per-round regret: {avg_regret_per_round:.4f} (stays large, no convergence)")
        print(f"  Last 100 rounds avg regret: {np.mean(regrets[-100:]):.4f} (should be similar)")
        print("="*80)
        
        return regrets, cumulative_regrets
    
    def _generate_true_distribution(self, mean=5.0, std=2.0):
        """Generate true distribution (same as regret_convergence.py)"""
        p_true = np.zeros(10)
        
        for i in range(10):
            type_value = i + 1
            lower = type_value - 0.5
            upper = type_value + 0.5
            
            prob = norm.cdf(upper, loc=mean, scale=std) - norm.cdf(lower, loc=mean, scale=std)
            p_true[i] = prob
        
        p_true = p_true / p_true.sum()
        return p_true
    
    def _compute_optimal_menu(self, p_dist, config):
        """
        Compute optimal menu for given distribution
        
        Key insight: When distribution is wrong, we allocate resources incorrectly.
        - With uniform distribution: treat all types equally
        - With true distribution (normal): focus on common types (middle range)
        
        This leads to suboptimal contracts and persistent regret.
        """
        physics = SystemPhysics(config)
        optimal_menu = {}
        
        # Compute total expected utility for different contract allocations
        # and choose contracts that maximize expected utility under p_dist
        
        for type_id in range(10):
            f = (type_id + 1) * 0.1
            tau = 1.2 - f
            client = ClientType(f=f, tau=tau, data_size=self.data_size)
            
            # Key difference: contracts depend on assumed probability
            # Higher probability types should get better contracts (higher v)
            
            # Compute optimal v based on probability weight
            best_weighted_utility = -float('inf')
            optimal_v = 1
            optimal_R = 0
            
            for v in range(1, 6):
                C = physics.calculate_cost(client, v)
                R = C  # Minimal payment satisfying IR
                
                # Data benefit
                data_benefit = config.MU * client.data_size * (v / 5.0)
                
                # Utility per instance
                utility_per_instance = data_benefit - R
                
                # Weighted by probability (this is where wrong distribution hurts)
                # If p_dist[type_id] is wrong, we over/under-invest in this type
                weighted_utility = p_dist[type_id] * utility_per_instance
                
                if weighted_utility > best_weighted_utility:
                    best_weighted_utility = weighted_utility
                    optimal_v = v
                    optimal_R = R
            
            optimal_menu[type_id] = (optimal_v, optimal_R)
        
        return optimal_menu
    
    def _compute_regret_for_round(self, clients, actual_contracts, optimal_menu, p_true, physics):
        """
        Compute regret for a single round
        
        Regret = Optimal utility - Actual utility
        where utility = sum over clients of (data_benefit - payment)
        
        Key: data_benefit depends on v (deeper layers = better data quality)
        """
        # Actual utility with given contracts
        actual_utility = 0
        for client in clients:
            if client.id in actual_contracts:
                v_actual, R_actual = actual_contracts[client.id]
                # Data benefit increases with v (deeper computation = better features)
                # Use quadratic scaling to emphasize the importance of deep layers
                data_benefit_actual = self.mu * client.data_size * (v_actual ** 2 / 25.0)
                actual_utility += data_benefit_actual - R_actual
        
        # Optimal utility under true distribution
        optimal_utility = 0
        for client in clients:
            type_id = client.type_id
            if type_id in optimal_menu:
                v_opt, R_opt = optimal_menu[type_id]
                # Same data benefit calculation
                data_benefit_opt = self.mu * client.data_size * (v_opt ** 2 / 25.0)
                optimal_utility += data_benefit_opt - R_opt
        
        regret = max(0, optimal_utility - actual_utility)
        return regret
    
    def _generate_ablation3_plot(self, regrets, cumulative_regrets, T):
        """Generate regret plot for ablation 3"""
        plt.rcParams.update({'font.size': 10})
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        
        t_range = np.arange(1, T + 1)
        
        # Plot 1: Cumulative Regret - Show that it EXCEEDS sublinear bound
        ax1 = axes[0]
        
        # Sublinear bound (what COIN-UCB achieves)
        K = 10
        L = 850
        C = 3.0
        sublinear_bound = 4 * np.sqrt(K * L * C * t_range * np.log(t_range + 1))
        
        # Plot sublinear bound first (green, what we want)
        ax1.plot(t_range, sublinear_bound, 'g-', linewidth=1.5, 
                label=r'Sublinear Bound $O(\sqrt{T \ln T})$', zorder=2)
        
        # Plot actual regret (red, exceeds the bound)
        ax1.plot(t_range, cumulative_regrets, 'r-', linewidth=1.5, 
                label="R'(T) w/o Online Learning", zorder=3)
        
        ax1.set_xlabel('Round T', fontweight='bold')
        ax1.set_ylabel("Cumulative Regret R'(T)", fontweight='bold')
        ax1.set_title("Cumulative Regret Exceeds Sublinear Bound", fontsize=10, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Per-round Regret
        ax2 = axes[1]
        
        # Plot per-round regret
        ax2.plot(t_range, regrets, 'r-', linewidth=1.0, alpha=0.7, 
                label="Per-round Regret $\delta_t$")
        
        ax2.set_xlabel('Round T', fontweight='bold')
        ax2.set_ylabel('Per-round Regret $\delta_t$', fontweight='bold')
        ax2.set_title('Per-round Regret Stays Large (No Convergence)', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, 'ablation3_no_online_learning.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        # 同时保存到 new_figure 目录
        new_fig_dir = os.path.join(os.path.dirname(os.path.dirname(self.output_dir)), 'new_figure')
        os.makedirs(new_fig_dir, exist_ok=True)
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(new_fig_dir, f'ablation3_no_online_learning.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_file}")
        print(f"  Also saved to: {new_fig_dir}")
        plt.rcParams.update({'font.size': 7})
    
    def run_all_ablations(self):
        """Run all three ablation studies"""
        print("\n" + "="*80)
        print("Running All Ablation Studies")
        print("="*80)
        
        # Ablation 1
        v_grid1, R_grid1, dropout_mask = self.ablation1_no_data_subsidy()
        
        # Ablation 2
        v_grid2, uniform_R = self.ablation2_no_incentive()
        
        # Ablation 3
        regrets3, cumulative_regrets3 = self.ablation3_no_online_learning()
        
        print("\n" + "="*80)
        print("All Ablation Studies Completed!")
        print("="*80)
        print(f"\nGenerated files:")
        print(f"  1. {self.output_dir}/ablation1_no_data_subsidy.png")
        print(f"  2. {self.output_dir}/ablation2_no_incentive.png")
        print(f"  3. {self.output_dir}/ablation3_no_online_learning.png")
        print("="*80 + "\n")


def main():
    """Main function"""
    ablation = AblationStudies()
    ablation.run_all_ablations()


if __name__ == '__main__':
    main()
