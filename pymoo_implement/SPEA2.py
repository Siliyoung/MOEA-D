from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize

# 定义问题
problem = get_problem("zdt1")

# 定义SPEA2算法
algorithm = SPEA2(
    pop_size=100,
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
