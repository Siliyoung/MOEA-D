from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 定义问题
problem = get_problem("zdt1")

# 定义MOGA算法
algorithm = GA(pop_size=100)

# 执行优化
res = minimize(
    problem,
    algorithm,
    termination=('n_gen', 500),
    seed=1,
    verbose=True
)

# 输出结果
Scatter().add(res.F).show()
