import numpy as np
from configs.config import RESNET_PROFILE

class SystemPhysics:
    def __init__(self, config):
        self.alpha = getattr(config, 'alpha', getattr(config, 'ALPHA', 5.0))
        self.beta = getattr(config, 'beta', getattr(config, 'BETA', 5.0))
        self.mu = getattr(config, 'mu', getattr(config, 'MU', 1.0))
        self.L = getattr(config, 'L', getattr(config, 'TOTAL_LAYERS', 5))
        self.profile = RESNET_PROFILE
        
        # 基准常数归一化
        self.BASE_W = 10.0
        self.BASE_D = 10.0
        # D_max = D(v=1)：早期切分点的传输量最大（符合ResNet结构）
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
        
        严格遵循论文公式，不添加额外机制
        """
        if v == 0:
            return 0.0
        
        W = self.get_W(v)
        D = self.get_D(v)
        
        # 论文原始公式
        utility = (self.alpha * W + 
                  self.beta * (self.D_max - D) + 
                  self.mu * client_type.data_size)
        
        return utility

    def _get_sorted_indices(self, all_types):
        # 按照 "效率" 排序 (Utility Ranking)
        # 效率定义：单位成本下的效用，这里简化为 f / tau (计算能力 / 通信延迟)
        efficiency = [(i, all_types[i].f / all_types[i].tau) for i in range(len(all_types))]
        efficiency.sort(key=lambda x: x[1])
        return np.array([x[0] for x in efficiency])

    def calculate_virtual_cost(self, target_k_idx, v, distribution_p, all_types):
        if v == 0: return 0.0
        sorted_indices = self._get_sorted_indices(all_types)
        
        # 找到目标类型在排序中的位置
        rank_pos = np.where(sorted_indices == target_k_idx)[0][0]

        type_k = all_types[target_k_idx]
        p_k = max(distribution_p[target_k_idx], 0.05)  # 使用更大的下界保护
        c_phy_k = self.calculate_cost(type_k, v)
        
        # 租金计算：遍历比 target_k 更强 (rank_pos 之后) 的类型
        rent_total = 0
        for m_pos in range(rank_pos + 1, len(sorted_indices)):
            m_idx = sorted_indices[m_pos]
            type_m = all_types[m_idx]
            p_m = distribution_p[m_idx]
            
            # IC: C_k(v) - C_m(v)
            # 弱者 k 的虚拟成本 = 物理成本 + Sum(强者 m 的租金)
            # 租金项 = (p_m / p_k) * (C_k(v) - C_m(v))
            
            delta_c = c_phy_k - self.calculate_cost(type_m, v)
            if delta_c > 0:
                rent_total += (p_m / p_k) * delta_c
                
        return c_phy_k + rent_total

    def solve_optimal_contract(self, distribution_p, all_types):
        # 严格遵循论文 Eq.(22) 和 Eq.(23)
        sorted_indices = self._get_sorted_indices(all_types)
        
        v_stars = {}
        # 1. 解耦求解每个类型的最优切分点 v*_k (Eq.22)
        for k_idx in range(len(all_types)):
            best_v = 1  # 从v=1开始，确保至少有一个切分点
            max_net_utility = -float('inf')
            
            # 遍历所有可能的切分点
            for v in range(1, self.L + 1):
                # 计算虚拟成本 C_k(v) (Eq.19)
                c_virtual = self.calculate_virtual_cost(k_idx, v, distribution_p, all_types)
                # 效用 = G(v) - C_k(v)
                g_v = self.calculate_server_utility(all_types[k_idx], v)
                net_utility = g_v - c_virtual
                
                if net_utility > max_net_utility:
                    max_net_utility = net_utility
                    best_v = v
            v_stars[k_idx] = best_v
            
        # 2. 基于切分点v计算报酬 R(v) (Eq.23的正确理解)
        # 核心逻辑：同一切分点v对应唯一最优报酬R(v)
        # 报酬是切分点的函数，而不是客户端的函数
        
        # 步骤2.1：确定全局最弱客户端（排序中第一个）
        global_weakest_idx = sorted_indices[0]
        
        # 步骤2.2：收集所有被选择的切分点
        selected_splits = set(v_stars.values())
        
        # 步骤2.3：为每个切分点v计算唯一报酬R(v)
        # 使用全局最弱客户端的成本作为基准，确保报酬单调递增
        R_by_v = {}
        for v in selected_splits:
            # 报酬 = 全局最弱客户端在该切分点的成本（紧IR约束）
            R_by_v[v] = self.calculate_cost(all_types[global_weakest_idx], v)
        
        # 步骤2.4：为每个客户端分配报酬（由其选择的切分点决定）
        rewards = {}
        for k_idx in range(len(all_types)):
            v_k = v_stars[k_idx]
            rewards[k_idx] = R_by_v[v_k]
        
        menu = []
        for i in range(len(all_types)):
            menu.append((v_stars[i], rewards[i]))
        return menu
