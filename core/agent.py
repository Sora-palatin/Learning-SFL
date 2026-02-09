import numpy as np

class OCDUCB_Agent:
    """
    OCD-UCB Agent - 严格按照论文第5章实现
    
    核心原理：
    1. 置信半径：Rad_k(t) = C * sqrt(ln(t) / N_k(t))
    2. 分布区间：[l_k, u_k] = [p_k - Rad_k, p_k + Rad_k]
    3. 乐观估计：宁可高估，不可低估
    4. 保证：乐观效用 >= 真实效用，后悔 delta >= 0
    """
    def __init__(self, num_types, exploration_c=0.5):
        self.K = num_types
        self.C = exploration_c  # 探索系数
        self.N_k = np.zeros(num_types)  # 每个类型的观测次数
        self.t = 0  # 当前时间步
        self.p_hat = np.full(num_types, 1.0 / num_types)  # 经验分布估计

    def calculate_confidence_radius(self):
        """
        计算置信半径 Rad_k(t)
        
        根据论文第5章：
        Rad_k(t) = C * sqrt(ln(t) / N_k(t))
        
        对于N_k=0的类型，设置最大半径以鼓励探索
        """
        if self.t == 0:
            return np.ones(self.K)
        
        Rad_k = np.zeros(self.K)
        for k in range(self.K):
            if self.N_k[k] == 0:
                Rad_k[k] = 1.0  # 未观测的类型，最大半径
            else:
                # 论文公式：Rad_k = C * sqrt(ln(t) / N_k)
                Rad_k[k] = self.C * np.sqrt(np.log(self.t + 1) / self.N_k[k])
        
        # 约束在[0, 1]区间
        return np.clip(Rad_k, 0, 1)

    def get_confidence_interval(self):
        """
        计算分布置信区间 [l_k, u_k]
        
        根据论文：
        l_k = max(0, p_hat_k - Rad_k)
        u_k = min(1, p_hat_k + Rad_k)
        
        这个区间用于约束乐观分布的取值范围
        """
        Rad_k = self.calculate_confidence_radius()
        
        # 计算经验分布
        if self.t > 0:
            self.p_hat = self.N_k / self.t
        else:
            self.p_hat = np.full(self.K, 1.0 / self.K)
        
        # 计算置信区间
        l_k = np.clip(self.p_hat - Rad_k, 0, 1)
        u_k = np.clip(self.p_hat + Rad_k, 0, 1)
        
        return l_k, u_k, Rad_k

    def get_optimistic_distribution(self):
        """
        计算乐观分布 p_opt
        
        UCB原理：宁可乐观高估，不可悲观低估
        
        策略：
        1. 使用置信区间的上界 u_k 作为乐观估计
        2. 这样可以高估高效用客户端的概率
        3. 由于Rad_k随时间递减，u_k会逐渐靠近p_hat
        4. 保证了乐观效用 >= 真实效用
        """
        l_k, u_k, Rad_k = self.get_confidence_interval()
        
        # 乐观估计：使用上界
        p_opt = u_k.copy()
        
        # 归一化（确保概率和为1）
        p_opt = p_opt / p_opt.sum()
        
        return p_opt

    def select_menu(self, physics_engine, all_types):
        """
        选择契约菜单
        
        流程：
        1. 计算乐观分布 p_opt
        2. 使用 p_opt 求解最优契约
        3. 返回契约菜单
        """
        # 获取乐观分布
        p_opt = self.get_optimistic_distribution()
        
        # 使用乐观分布求解契约
        menu = physics_engine.solve_optimal_contract(p_opt, all_types)
        
        return menu

    def update(self, observed_type_idx):
        """
        更新观测统计
        
        参数：
            observed_type_idx: 当前时间步观测到的客户端类型索引
        """
        self.N_k[observed_type_idx] += 1
        self.t += 1
        
        # 更新经验分布
        self.p_hat = self.N_k / self.t

    def get_radius(self):
        """返回当前的平均置信半径"""
        Rad_k = self.calculate_confidence_radius()
        return np.mean(Rad_k)
