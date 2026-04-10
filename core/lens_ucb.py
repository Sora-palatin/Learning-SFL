"""
LENS-UCB (Learning with Exploration and iNcentive for Split-learning via Upper Confidence Bound) Learner

"""
import numpy as np
from core.regret import calculate_instant_regret


class LENS_UCB_Learner:
 """LENS-UCB"""
    
    def __init__(self, physics, client_types, true_distribution, T, C=0.5):
        """
 LENS-UCB
        
        Args:
 physics: SystemPhysics
 client_types:
 true_distribution: （）
 T:
 C:
        """
        self.physics = physics
        self.client_types = client_types
        self.true_distribution = true_distribution
        self.T = T
        self.C = C
        
        self.K = len(client_types)
        self.t = 0
        
 # 
        self.N_k = np.zeros(self.K)  # sample count per client type
        self.cumulative_regret = 0.0
        self.instant_regret_history = []
        
 # 
        self.optimal_menu = physics.solve_optimal_contract(
            true_distribution, client_types
        )
    
    def confidence_radius(self, k, t):
        """

        
        Args:
 k:
 t:
        
        Returns:

        """
        if self.N_k[k] == 0:
            return float('inf')
        
        return self.C * np.sqrt(np.log(t) / self.N_k[k])
    
    def get_optimistic_distribution(self):
        """

        
        Returns:
 p_opt
        """
        if self.t == 0:
            return np.ones(self.K) / self.K
        
 # 
        p_emp = self.N_k / max(self.t, 1)
        
 # p_opt = p_emp + radius
        p_opt = np.zeros(self.K)
        for k in range(self.K):
            radius = self.confidence_radius(k, self.t)
            p_opt[k] = min(1.0, p_emp[k] + radius)
        
 # 
        p_opt = p_opt / np.sum(p_opt)
        
        return p_opt
    
    def get_current_contract(self):
        """
 （）
        
        Returns:

        """
        p_opt = self.get_optimistic_distribution()
        return self.physics.solve_optimal_contract(p_opt, self.client_types)
    
    def step(self):
 """"""
        self.t += 1
        
 # 
        k_t = np.random.choice(self.K, p=self.true_distribution)
        
 # 
        self.N_k[k_t] += 1
        
 # 
        current_menu = self.get_current_contract()
        
 # 
        instant_regret = calculate_instant_regret(
            self.physics,
            self.client_types,
            k_t,
            self.optimal_menu,
            current_menu,
            self.t,
            self.T,
            add_noise=False  # simulation test: no noise injection
        )
        
 # 
        self.cumulative_regret += instant_regret
        self.instant_regret_history.append(instant_regret)
        
        return k_t, instant_regret
    
    def get_statistics(self):
        """

        
        Returns:

        """
        return {
            'cumulative_regret': self.cumulative_regret,
            'regret_rate': self.cumulative_regret / max(self.t, 1),
            'N_k': self.N_k.copy(),
            'empirical_distribution': self.N_k / max(self.t, 1),
            'optimistic_distribution': self.get_optimistic_distribution()
        }
