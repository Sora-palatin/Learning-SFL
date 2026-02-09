"""
OCD-UCB (Optimistic Contract Design with Upper Confidence Bound) Learner
用于未知分布下的在线契约学习
"""
import numpy as np
from core.regret import calculate_instant_regret


class OCD_UCB_Learner:
    """OCD-UCB学习器"""
    
    def __init__(self, physics, client_types, true_distribution, T, C=0.5):
        """
        初始化OCD-UCB学习器
        
        Args:
            physics: SystemPhysics对象
            client_types: 客户端类型列表
            true_distribution: 真实分布（用于计算后悔值）
            T: 总时间步数
            C: 置信参数
        """
        self.physics = physics
        self.client_types = client_types
        self.true_distribution = true_distribution
        self.T = T
        self.C = C
        
        self.K = len(client_types)
        self.t = 0
        
        # 统计信息
        self.N_k = np.zeros(self.K)  # 每个类型被采样的次数
        self.cumulative_regret = 0.0
        self.instant_regret_history = []
        
        # 计算最优契约（用于后悔值计算）
        self.optimal_menu = physics.solve_optimal_contract(
            true_distribution, client_types
        )
    
    def confidence_radius(self, k, t):
        """
        计算置信半径
        
        Args:
            k: 客户端类型索引
            t: 当前时间步
        
        Returns:
            置信半径
        """
        if self.N_k[k] == 0:
            return float('inf')
        
        return self.C * np.sqrt(np.log(t) / self.N_k[k])
    
    def get_optimistic_distribution(self):
        """
        获取乐观分布估计
        
        Returns:
            乐观分布 p_opt
        """
        if self.t == 0:
            return np.ones(self.K) / self.K
        
        # 经验分布
        p_emp = self.N_k / max(self.t, 1)
        
        # 乐观分布：p_opt = p_emp + radius
        p_opt = np.zeros(self.K)
        for k in range(self.K):
            radius = self.confidence_radius(k, self.t)
            p_opt[k] = min(1.0, p_emp[k] + radius)
        
        # 归一化
        p_opt = p_opt / np.sum(p_opt)
        
        return p_opt
    
    def get_current_contract(self):
        """
        获取当前契约（基于乐观分布）
        
        Returns:
            当前契约菜单
        """
        p_opt = self.get_optimistic_distribution()
        return self.physics.solve_optimal_contract(p_opt, self.client_types)
    
    def step(self):
        """执行一步学习"""
        self.t += 1
        
        # 从真实分布中采样客户端类型
        k_t = np.random.choice(self.K, p=self.true_distribution)
        
        # 更新统计
        self.N_k[k_t] += 1
        
        # 获取当前契约
        current_menu = self.get_current_contract()
        
        # 计算瞬时后悔
        instant_regret = calculate_instant_regret(
            self.physics,
            self.client_types,
            k_t,
            self.optimal_menu,
            current_menu,
            self.t,
            self.T,
            add_noise=False  # 仿真测试不添加噪声
        )
        
        # 更新累计后悔
        self.cumulative_regret += instant_regret
        self.instant_regret_history.append(instant_regret)
        
        return k_t, instant_regret
    
    def get_statistics(self):
        """
        获取学习统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'cumulative_regret': self.cumulative_regret,
            'regret_rate': self.cumulative_regret / max(self.t, 1),
            'N_k': self.N_k.copy(),
            'empirical_distribution': self.N_k / max(self.t, 1),
            'optimistic_distribution': self.get_optimistic_distribution()
        }
