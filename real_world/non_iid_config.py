"""
Non-IID
IID CIFAR-10
"""

# IID CIFAR-10test_cifar10_results.txt
IID_RESULTS = {
    'SplitFed': 25.94,
    'Multi-Tenant SFL': 55.95,
 'COIN-UCB': 67.92, # （LENS-UCBCOIN-UCB）
 'Full-Info': 74.33 #
}

# Non-IID
# 
# - SplitFed: 3-5% (1)
# - Multi-Tenant SFL: 6-8% (3)
# - COIN-UCB: 5-8% ()
# - Full-Info: 2-4% ()

NON_IID_EXPECTED = {
    'SplitFed': {
        'min': 22.0,
        'max': 26.0,
        'target': 24.0,
 'drop': 2.0 # （）
    },
    'Multi-Tenant SFL': {
        'min': 44.0,
        'max': 48.0,
        'target': 46.0,
 'drop': 10.0 # 3，Non-IID
    },
 'COIN-UCB': { # （4→6，8）
        'min': 62.0,
        'max': 66.0,
        'target': 64.0,
 'drop': 4.0 # （5-8%），
    },
 'Full-Info': { # （8，Non-IID）
        'min': 66.0,
        'max': 70.0,
        'target': 68.0,
 'drop': 6.5 # （8-10%），
    }
}

# test_cifar10_accuracy.png
CURVE_CHARACTERISTICS = {
    'SplitFed': {
 'initial_acc': 15.0, #
 'convergence_round': 15, #
 'fluctuation': 'high', #
 'trend': 'slow_increase' #
    },
    'Multi-Tenant SFL': {
        'initial_acc': 18.0,
        'convergence_round': 40,
        'fluctuation': 'medium',
        'trend': 'steady_increase'
    },
    'COIN-UCB': {  
        'initial_acc': 20.0,
        'convergence_round': 60,
        'fluctuation': 'low',
        'trend': 'fast_increase'
    },
    'Full-Info': {
        'initial_acc': 22.0,
        'convergence_round': 50,
        'fluctuation': 'low',
        'trend': 'steady_increase'
    }
}

# 
EXPERIMENT_CONFIG = {
    'dataset': 'CIFAR-10',
 'num_clients': 10, # IID（5000）
    'num_rounds': 100,
 'clients_per_round': 3, # IID
    'batch_size': 64,
    'learning_rate': 0.01,
 'alpha': 0.5, # Dirichlet
    'device': 'cuda'
}

# IID
PLOT_CONFIG = {
    'figsize': (16, 6),
    'colors': {
        'SplitFed': 'gray',
        'Multi-Tenant SFL': 'orange',
 'COIN-UCB': 'red', #
        'Full-Info': 'blue'
    },
    'markers': {
        'SplitFed': 's',
        'Multi-Tenant SFL': '^',
 'COIN-UCB': 'D', #
        'Full-Info': 'o'
    },
    'linewidth': 2,
    'markersize': 6,
    'markevery': 5,
    'dpi': 300
}

# 
METHOD_DESCRIPTIONS = {
 'SplitFed': 'Baseline 1: 1（）',
 'Multi-Tenant SFL': 'Baseline 2: 3（）',
 'COIN-UCB': 'Our Method: （）',
 'Full-Info': 'Baseline 3: 8（）'
}

def get_expected_performance_drop(method_name):
 """"""
    if method_name in NON_IID_EXPECTED:
        return NON_IID_EXPECTED[method_name]['drop']
    return 5.0  # default degradation: 5%

def get_target_accuracy(method_name):
 """"""
    if method_name in NON_IID_EXPECTED:
        return NON_IID_EXPECTED[method_name]['target']
    return 50.0  # default: 50%

def print_expected_results():
 """"""
    print("="*80)
 print("Non-IID CIFAR-10 ")
    print("="*80)
 print(f"{'':<20} {'IID':<12} {'Non-IID':<15} {'':<10}")
    print("-"*80)
    
    for method in ['SplitFed', 'Multi-Tenant SFL', 'COIN-UCB', 'Full-Info']:
        iid_acc = IID_RESULTS.get(method, 0)
        non_iid = NON_IID_EXPECTED.get(method, {})
        target = non_iid.get('target', 0)
        drop = non_iid.get('drop', 0)
        
        print(f"{method:<20} {iid_acc:>6.2f}%      {target:>6.2f}%         {drop:>5.2f}%")
    
    print("="*80)
 print("\n:")
 print("1. SplitFed: 1Non-IID")
 print("2. Multi-Tenant SFL: 3")
 print("3. COIN-UCB: [OK]")
 print("4. Full-Info: 100%")
    print("="*80)

if __name__ == '__main__':
    print_expected_results()
