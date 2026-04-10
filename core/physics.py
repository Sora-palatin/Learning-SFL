import numpy as np
from configs.config import RESNET_PROFILE

class SystemPhysics:
    def __init__(self, config):
        self.alpha = getattr(config, 'alpha', getattr(config, 'ALPHA', 5.0))
        self.beta = getattr(config, 'beta', getattr(config, 'BETA', 5.0))
        self.mu = getattr(config, 'mu', getattr(config, 'MU', 1.0))
        self.L = getattr(config, 'L', getattr(config, 'TOTAL_LAYERS', 5))
        self.profile = RESNET_PROFILE
        
 # 
        self.BASE_W = 10.0
        self.BASE_D = 10.0
 # D_max = D(v=1)ResNet
        self.D_max = self.get_D(1)

    def get_W(self, v):
        if v == 0: return 0.0
        return self.profile[v]['W'] * self.BASE_W

    def get_D(self, v):
        if v == 0: return 0.0
        return self.profile[v]['D'] * self.BASE_D

    def calculate_cost(self, client_type, v):
        if v == 0: return 0.0
        W = self.get_W(v)
        D = self.get_D(v)
        return W / client_type.f + D * client_type.tau

    def calculate_server_utility(self, client_type, v):
        """
        Server utility for assigning task v to client k.
        G(k,v) = alpha*W(v) + beta*(D_max - D(v)) + mu*data_size
        
 ，
        """
        if v == 0:
            return 0.0
        
        W = self.get_W(v)
        D = self.get_D(v)
        
 # 
        utility = (self.alpha * W + 
                  self.beta * (self.D_max - D) + 
                  self.mu * client_type.data_size)
        
        return utility

    def _get_sorted_indices(self, all_types):
 # "" (Utility Ranking)
 # f / tau ( / )
        efficiency = [(i, all_types[i].f / all_types[i].tau) for i in range(len(all_types))]
        efficiency.sort(key=lambda x: x[1])
        return np.array([x[0] for x in efficiency])

    def calculate_virtual_cost(self, target_k_idx, v, distribution_p, all_types):
        if v == 0: return 0.0
        sorted_indices = self._get_sorted_indices(all_types)
        
 # 
        rank_pos = np.where(sorted_indices == target_k_idx)[0][0]

        type_k = all_types[target_k_idx]
        p_k = max(distribution_p[target_k_idx], 0.05)  # apply lower-bound clipping for numerical stability
        c_phy_k = self.calculate_cost(type_k, v)
        
 # target_k (rank_pos ) 
        rent_total = 0
        for m_pos in range(rank_pos + 1, len(sorted_indices)):
            m_idx = sorted_indices[m_pos]
            type_m = all_types[m_idx]
            p_m = distribution_p[m_idx]
            
            # IC: C_k(v) - C_m(v)
 # k = + Sum( m )
 # = (p_m / p_k) * (C_k(v) - C_m(v))
            
            delta_c = c_phy_k - self.calculate_cost(type_m, v)
            if delta_c > 0:
                rent_total += (p_m / p_k) * delta_c
                
        return c_phy_k + rent_total

    def solve_optimal_contract(self, distribution_p, all_types):
 # Eq.(22) Eq.(23)
        sorted_indices = self._get_sorted_indices(all_types)
        
        v_stars = {}
 # 1. v*_k (Eq.22)
        for k_idx in range(len(all_types)):
            best_v = 1  # start from v=1 to guarantee at least one valid cut layer
            max_net_utility = -float('inf')
            
 # 
            for v in range(1, self.L + 1):
 # C_k(v) (Eq.19)
                c_virtual = self.calculate_virtual_cost(k_idx, v, distribution_p, all_types)
 # = G(v) - C_k(v)
                g_v = self.calculate_server_utility(all_types[k_idx], v)
                net_utility = g_v - c_virtual
                
                if net_utility > max_net_utility:
                    max_net_utility = net_utility
                    best_v = v
            v_stars[k_idx] = best_v
            
 # 2. v R(v) (Eq.23)
 # vR(v)
 # 
        
 # 2.1
        global_weakest_idx = sorted_indices[0]
        
 # 2.2
        selected_splits = set(v_stars.values())
        
 # 2.3vR(v)
 # 
        R_by_v = {}
        for v in selected_splits:
 # = IR
            R_by_v[v] = self.calculate_cost(all_types[global_weakest_idx], v)
        
 # 2.4
        rewards = {}
        for k_idx in range(len(all_types)):
            v_k = v_stars[k_idx]
            rewards[k_idx] = R_by_v[v_k]
        
        menu = []
        for i in range(len(all_types)):
            menu.append((v_stars[i], rewards[i]))
        return menu
