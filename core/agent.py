import numpy as np

class LENSUCB_Agent:
    """
 LENS-UCB Agent - 5
    
 ：
 1. ：Rad_k(t) = C * sqrt(ln(t) / N_k(t))
 2. ：[l_k, u_k] = [p_k - Rad_k, p_k + Rad_k]
 3. ：，
 4. ： >= ， delta >= 0
    """
    def __init__(self, num_types, exploration_c=0.5):
        self.K = num_types
        self.C = exploration_c  # exploration coefficient
        self.N_k = np.zeros(num_types)  # observation count per type
        self.t = 0  # current time step
        self.p_hat = np.full(num_types, 1.0 / num_types)  # empirical distribution estimate

    def calculate_confidence_radius(self):
        """
 Rad_k(t)
        
 5：
        Rad_k(t) = C * sqrt(ln(t) / N_k(t))
        
 N_k=0，
        """
        if self.t == 0:
            return np.ones(self.K)
        
        Rad_k = np.zeros(self.K)
        for k in range(self.K):
            if self.N_k[k] == 0:
                Rad_k[k] = 1.0  # unobserved type, maximum radius
            else:
 # Rad_k = C * sqrt(ln(t) / N_k)
                Rad_k[k] = self.C * np.sqrt(np.log(self.t + 1) / self.N_k[k])
        
 # [0, 1]
        return np.clip(Rad_k, 0, 1)

    def get_confidence_interval(self):
        """
 [l_k, u_k]
        
 ：
        l_k = max(0, p_hat_k - Rad_k)
        u_k = min(1, p_hat_k + Rad_k)
        

        """
        Rad_k = self.calculate_confidence_radius()
        
 # 
        if self.t > 0:
            self.p_hat = self.N_k / self.t
        else:
            self.p_hat = np.full(self.K, 1.0 / self.K)
        
 # 
        l_k = np.clip(self.p_hat - Rad_k, 0, 1)
        u_k = np.clip(self.p_hat + Rad_k, 0, 1)
        
        return l_k, u_k, Rad_k

    def get_optimistic_distribution(self):
        """
 p_opt
        
 UCB：，
        
 ：
 1. u_k
 2.
 3. Rad_k，u_kp_hat
 4. >=
        """
        l_k, u_k, Rad_k = self.get_confidence_interval()
        
 # 
        p_opt = u_k.copy()
        
 # 1
        p_opt = p_opt / p_opt.sum()
        
        return p_opt

    def select_menu(self, physics_engine, all_types):
        """

        
 ：
 1. p_opt
 2. p_opt
 3.
        """
 # 
        p_opt = self.get_optimistic_distribution()
        
 # 
        menu = physics_engine.solve_optimal_contract(p_opt, all_types)
        
        return menu

    def update(self, observed_type_idx):
        """

        
 ：
 observed_type_idx:
        """
        self.N_k[observed_type_idx] += 1
        self.t += 1
        
 # 
        self.p_hat = self.N_k / self.t

    def get_radius(self):
 """"""
        Rad_k = self.calculate_confidence_radius()
        return np.mean(Rad_k)
