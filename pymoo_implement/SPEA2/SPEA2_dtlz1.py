from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 定义问题
problem = get_problem("dtlz1", n_obj=3, n_var=7)  # 根据需要设置目标数和变量数

# 定义SPEA2算法
algorithm = SPEA2(
    pop_size=100,
    eliminate_duplicates=True
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

# # 输出结果
# print("Hypervolume:", res.F)



# 获取zdt问题的真实Pareto前沿
pareto_front = problem.pareto_front(n_pareto_points=250)

# 输出结果并绘制三维散点图
plot = Scatter(legend=True, labels=["F1", "F2", "F3"])
# 可视化：绘制优化解
plot.add(pareto_front, color="blue", alpha=0.2, s=50, label="True Pareto Front")
plot.add(res.F, color="red", s=10, label="MOEA/D Results")
plot.show()
