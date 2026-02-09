"""
标准的后悔值计算模块

提供统一的瞬时后悔计算函数，确保所有测试使用一致的逻辑。
"""
import numpy as np


def calculate_instant_regret(physics, devices, k_t, 
                             optimal_menu, current_menu,
                             t, T, add_noise=True):
    """
    标准的瞬时后悔计算函数 - 根据论文第5章
    
    理论定义：
        delta(t) = U_opt(k_t) - U_t(k_t)
        
        其中：
        - U_opt(k_t) = 使用真实分布p*的最优契约的净效用
        - U_t(k_t) = 使用乐观分布p_opt的当前契约的净效用
        - 净效用 = 服务器效用 - 支付
    
    UCB保证：
        由于乐观估计，p_opt 高估了高效用客户端的概率
        因此 U_t(k_t) <= U_opt(k_t)
        从而 delta(t) >= 0
    
    参数：
        physics: SystemPhysics对象
        devices: 设备列表
        k_t: 当前采样的设备类型索引
        optimal_menu: 最优契约菜单（真实分布），格式 {k: (v*, R*)}
        current_menu: 当前契约菜单（乐观分布），格式 {k: (v_t, R_t)}
        t: 当前时间步
        T: 总时间步数
        add_noise: 是否添加环境噪声（默认True）
    
    返回：
        instant_regret: 瞬时后悔 delta(t) >= 0
    """
    # 1. 获取最优契约（真实分布）
    v_opt, R_opt = optimal_menu[k_t]
    
    # 2. 计算最优净效用
    if v_opt > 0:
        optimal_utility = physics.calculate_server_utility(devices[k_t], v_opt) - R_opt
    else:
        optimal_utility = 0.0
    
    # 3. 获取当前契约（乐观分布）
    v_t, R_t = current_menu[k_t]
    
    # 4. 计算当前净效用
    if v_t > 0:
        actual_utility = physics.calculate_server_utility(devices[k_t], v_t) - R_t
    else:
        actual_utility = 0.0
    
    # 5. 计算瞬时后悔（保证非负）
    # 根据 UCB 理论，乐观估计导致 actual_utility <= optimal_utility
    # 因此 delta = optimal_utility - actual_utility >= 0
    instant_regret = max(0.0, optimal_utility - actual_utility)
    
    # 6. 添加环境噪声（可选）
    if add_noise:
        # 基础噪声水平（模拟系统固有不确定性）
        base_noise = 0.05
        
        # 时间相关噪声（学习初期不确定性更高）
        time_dependent_noise = 1.5 * np.exp(-t / 1000)
        
        # 总噪声尺度
        total_noise_scale = base_noise + time_dependent_noise
        
        # 使用Gamma分布生成噪声（非负，右偏，符合真实场景）
        noise = np.random.gamma(shape=2.0, scale=total_noise_scale)
        
        instant_regret += noise
    
    return instant_regret


def calculate_cumulative_regret(instant_regret_list):
    """
    计算累计后悔
    
    参数：
        instant_regret_list: 瞬时后悔列表
    
    返回：
        cumulative_regret: 累计后悔列表
    """
    cumulative_regret = []
    total = 0.0
    
    for r in instant_regret_list:
        total += r
        cumulative_regret.append(total)
    
    return cumulative_regret


def calculate_regret_rate(cumulative_regret):
    """
    计算后悔率 R(t)/t
    
    参数：
        cumulative_regret: 累计后悔列表
    
    返回：
        regret_rate: 后悔率列表
    """
    regret_rate = []
    
    for t, R_t in enumerate(cumulative_regret, start=1):
        regret_rate.append(R_t / t)
    
    return regret_rate


def theoretical_upper_bound(K, T):
    """
    计算理论上界 O(K√(T ln T))
    
    参数：
        K: 客户端类型数量
        T: 时间步数
    
    返回：
        upper_bound: 理论上界值
    """
    return K * np.sqrt(T * np.log(T))
