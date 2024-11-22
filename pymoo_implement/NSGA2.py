from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize

# 定义问题
problem = get_problem("zdt1")

# 定义NSGA-II算法
algorithm = NSGA2(
    pop_size=100,
    sampling="real_random",
    crossover="real_sbx",
    mutation="real_pm",
    eliminate_duplicates=True
)

# 执行优化
res = minimize(
    problem,
    algorithm,
    termination=('n_gen', 500),
    seed=1,
    save_history=True,
    verbose=True
)

# 输出结果
print("Hypervolume:", res.F)
