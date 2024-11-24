import numpy as np
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# 示例问题
problem = get_problem("zdt1")

# 初始化粒子群
n_particles = 50
n_dimensions = problem.n_var
max_generations = 100

X = np.random.rand(n_particles, n_dimensions)
V = np.random.rand(n_particles, n_dimensions) * 0.1

# 记录全局 Pareto 前沿
global_archive = []

for gen in range(max_generations):
    # 计算目标值
    F = problem.evaluate(X)
    
    # 更新非支配解
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    current_pareto = X[nds]
    global_archive.append(current_pareto)
    
    # 粒子速度更新规则
    for i in range(n_particles):
        V[i] = 0.7 * V[i] + 0.2 * np.random.rand() * (current_pareto[0] - X[i])
        X[i] += V[i]
    
    # 粒子边界控制
    X = np.clip(X, problem.xl, problem.xu)

print("okk")

