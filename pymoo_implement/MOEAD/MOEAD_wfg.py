from pymoo.algorithms.moo.moead import MOEAD
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize

# 定义问题，选择一个WFG系列问题，例如 WFG1
n_var = 12  # 假设问题有 12 个决策变量
problem = get_problem("wfg1", n_obj=3, n_var=n_var, k=4)  # 目标数和变量数

# 获取WFG问题的真实Pareto前沿
pareto_front = problem.pareto_front(n_pareto_points=250)

# 获取参考方向
ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_points=91)  # 2目标问题，99个参考方向

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
    termination=('n_gen', 250),  # 250代终止
    seed=1,
    save_history=True,           # 记录优化历史
    verbose=True
)

# 可视化：绘制真实 Pareto 前沿和优化解
plot = Scatter(legend=True, labels=["F1", "F2","F3"])
plot.add(pareto_front, color="blue", alpha=0.2, s=50, label="True Pareto Front")
plot.add(res.F, color="red", s=10, label="MOEA/D Results")
plot.show()
