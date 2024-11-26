from pymoo.algorithms.moo.moead import MOEAD
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter


from pymoo.optimize import minimize

# 定义问题,获取一个标准测试问题,zdt1 是一个两目标的连续优化问题
problem = get_problem("zdt1")

# 切比雪夫分解法的参考方向
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)

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
    termination=('n_gen', 500),  # 500代终止
    seed=1,
    save_history=True,           # 记录优化历史
    verbose=True
)

# 输出结果
#print("Hypervolume:", res.F)

Scatter().add(res.F).show()      # 绘制前沿解