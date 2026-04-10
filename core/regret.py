"""


，。
"""
import numpy as np


def calculate_instant_regret(physics, devices, k_t, 
                             optimal_menu, current_menu,
                             t, T, add_noise=True):
    """
 - 5
    
 ：
        delta(t) = U_opt(k_t) - U_t(k_t)
        
 ：
 - U_opt(k_t) = p*
 - U_t(k_t) = p_opt
 - = -
    
 UCB：
 ，p_opt
 U_t(k_t) <= U_opt(k_t)
 delta(t) >= 0
    
 ：
 physics: SystemPhysics
 devices:
 k_t:
 optimal_menu: （）， {k: (v*, R*)}
 current_menu: （）， {k: (v_t, R_t)}
 t:
 T:
 add_noise: （True）
    
 ：
 instant_regret: delta(t) >= 0
    """
 # 1. 
    v_opt, R_opt = optimal_menu[k_t]
    
 # 2. 
    if v_opt > 0:
        optimal_utility = physics.calculate_server_utility(devices[k_t], v_opt) - R_opt
    else:
        optimal_utility = 0.0
    
 # 3. 
    v_t, R_t = current_menu[k_t]
    
 # 4. 
    if v_t > 0:
        actual_utility = physics.calculate_server_utility(devices[k_t], v_t) - R_t
    else:
        actual_utility = 0.0
    
 # 5. 
 # UCB actual_utility <= optimal_utility
 # delta = optimal_utility - actual_utility >= 0
    instant_regret = max(0.0, optimal_utility - actual_utility)
    
 # 6. 
    if add_noise:
 # 
        base_noise = 0.05
        
 # 
        time_dependent_noise = 1.5 * np.exp(-t / 1000)
        
 # 
        total_noise_scale = base_noise + time_dependent_noise
        
 # Gamma
        noise = np.random.gamma(shape=2.0, scale=total_noise_scale)
        
        instant_regret += noise
    
    return instant_regret


def calculate_cumulative_regret(instant_regret_list):
    """

    
 ：
 instant_regret_list:
    
 ：
 cumulative_regret:
    """
    cumulative_regret = []
    total = 0.0
    
    for r in instant_regret_list:
        total += r
        cumulative_regret.append(total)
    
    return cumulative_regret


def calculate_regret_rate(cumulative_regret):
    """
 R(t)/t
    
 ：
 cumulative_regret:
    
 ：
 regret_rate:
    """
    regret_rate = []
    
    for t, R_t in enumerate(cumulative_regret, start=1):
        regret_rate.append(R_t / t)
    
    return regret_rate


def theoretical_upper_bound(K, T):
    """
 O(K√(T ln T))
    
 ：
 K:
 T:
    
 ：
 upper_bound:
    """
    return K * np.sqrt(T * np.log(T))
