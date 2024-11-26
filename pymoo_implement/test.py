from pymoo.algorithms.moo.moead import MOEAD
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

# 定义问题 zdt1, dtlz1, wfg1
problem = get_problem("dtlz7")

# 切比雪夫分解法的参考方向
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)

# 定义MOEA/D算法
algorithm = MOEAD(
    ref_dirs=ref_dirs,
    n_neighbors=15,
    prob_neighbor_mating=0.7
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

# 使用 Scatter 绘制 Pareto 前沿，同时结合 matplotlib 添加标题
Scatter().add(res.F).show()

