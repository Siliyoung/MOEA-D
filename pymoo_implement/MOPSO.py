from pymoo.algorithms.moo.mopso import MOPSO
from pymoo.factory import get_problem
from pymoo.optimize import minimize

# 定义问题
problem = get_problem("zdt1")

# 定义MOPSO算法
algorithm = MOPSO(
    pop_size=100,
    seed=1,
    verbose=True
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
