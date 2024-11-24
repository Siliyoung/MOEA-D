from pymoo.algorithms.moo.mopso import MOPSO
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 选择测试问题：ZDT系列、DTLZ系列或WFG系列问题
problem = get_problem("zdt1")  # 可以替换为 "dtlz1" 或 "wfg1"

# 定义MOPSO算法
algorithm = MOPSO(
    pop_size=100,  # 种群大小
    max_velocity=2.0,  # 最大速度
    w=0.5,  # 惯性权重
    c1=1.5,  # 自我认知因子
    c2=1.5,  # 群体认知因子
    verbose=True
)

# 执行优化
res = minimize(
    problem,
    algorithm,
    termination=('n_gen', 250),  # 250代终止
    seed=1,
    save_history=True,
    verbose=True
)

# 输出结果
print("Hypervolume:", res.F)

# 可视化Pareto前沿
Scatter().add(res.F).show()
