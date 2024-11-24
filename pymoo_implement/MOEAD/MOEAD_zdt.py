from pymoo.algorithms.moo.moead import MOEAD
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize

# 定义ZDT问题
problem = get_problem("zdt1")  # 可换为 "zdt2", "zdt3", 等


# 获取zdt问题的真实Pareto前沿
pareto_front = problem.pareto_front(n_pareto_points=250)

# 定义参考方向
ref_dirs = get_reference_directions("das-dennis", n_dim=2, n_points=99)

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

# 可视化：绘制优化解
plot = Scatter(legend=True, labels=["F1", "F2"])
plot.add(pareto_front, color="blue", alpha=0.2, s=50, label="True Pareto Front")
plot.add(res.F, color="red", s=10, label="MOEA/D Results")
plot.show()

