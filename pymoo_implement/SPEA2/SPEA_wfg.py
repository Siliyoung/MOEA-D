from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 定义问题，这里以 wfg1 为例
problem = get_problem("wfg1", n_obj=3, n_var=12)  # 3个目标，12个决策变量

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

pareto_front = problem.pareto_front(n_pareto_points=250)

# 可视化：绘制真实 Pareto 前沿和优化解
plot = Scatter(legend=True, labels=["F1", "F2","F3"])
plot.add(pareto_front, color="blue", alpha=0.2, s=30, label="True Pareto Front")
plot.add(res.F, color="red", s=10, label="SPEA2 Results")
plot.show()
