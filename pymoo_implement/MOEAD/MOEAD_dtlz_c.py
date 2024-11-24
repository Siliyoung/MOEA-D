from pymoo.algorithms.moo.moead import MOEAD
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize

# 定义问题
problem = get_problem("dtlz2", n_obj=3)

# 生成参考方向
ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_partitions=12)

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
    termination=('n_gen', 250),
    seed=1,
    save_history=True,
    verbose=True
)

# 获取真实 Pareto 前沿
pareto_front = problem.pareto_front()

# 可视化：绘制真实 Pareto 前沿和算法结果
plot = Scatter(legend=True, labels=["F1", "F2", "F3"])
plot.add(pareto_front, color="blue", label="True Pareto Front")
plot.add(res.F, color="red", label="Algorithm Result")
plot.show()
