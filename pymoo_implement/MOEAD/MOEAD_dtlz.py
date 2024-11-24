from pymoo.algorithms.moo.moead import MOEAD
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize

# # 定义DTLZ1问题
# problem = get_problem("dtlz1", n_obj=3)

# # 定义DTLZ2问题
# problem = get_problem("dtlz2", n_obj=3)

# # 定义DTLZ3问题
# problem = get_problem("dtlz3", n_obj=3)

# # 定义DTLZ4问题
# problem = get_problem("dtlz4", n_obj=3)

# # 定义DTLZ5问题
# problem = get_problem("dtlz5", n_obj=3)

# # 定义DTLZ6问题
# problem = get_problem("dtlz6", n_obj=3)

# # 定义DTLZ7问题
problem = get_problem("dtlz1", n_obj=3)



# 生成参考方向
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

# 定义MOEA/D算法
algorithm = MOEAD(
    ref_dirs,
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

# 获取问题的真实Pareto前沿
pareto_front = problem.pareto_front()

# 输出结果并绘制三维散点图
plot = Scatter(legend=True, labels=["F1", "F2", "F3"])
# 可视化：绘制优化解
plot.add(pareto_front, color="blue", alpha=0.2, s=50, label="True Pareto Front")
plot.add(res.F, color="red", s=10, label="MOEA/D Results")
plot.show()
