import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

# 获取WFG3问题，n_obj=4，确保 n_var 是合适的
problem = get_problem("wfg3", n_obj=4, n_var=20)

# 生成参考方向
ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)

# 获取真实的Pareto前沿，手动指定参考方向
pareto_front = problem.pareto_front(ref_dirs)

# 定义NSGA-II算法
algorithm = NSGA2(pop_size=100,
                  sampling=FloatRandomSampling(),
                  crossover=SimulatedBinaryCrossover(),
                  mutation=PolynomialMutation(),
                  eliminate_duplicates=True)

# 执行优化
res = minimize(problem,
               algorithm,
               ('n_gen', 250),
               seed=1,
               verbose=False)

# 绘制 3D 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制优化结果
ax.scatter(res.F[:, 0], res.F[:, 1], res.F[:, 2], c="red", label="NSGA-II Results")

# 绘制真实 Pareto 前沿（如果维度少于 3，可以选择前 3 个目标）
ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], c="blue", label="True Pareto Front")

# 设置标签
ax.set_xlabel('F1')
ax.set_ylabel('F2')
ax.set_zlabel('F3')

ax.legend()
plt.show()
