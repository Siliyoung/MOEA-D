from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions


# 获取WFG1问题 (可以选择其他WFG问题)
problem = get_problem("wfg1", n_obj=3, n_var=12)

# problem = get_problem("wfg2", n_obj=3, n_var=12)


# 生成参考方向
ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)

# 获取真实的Pareto前沿
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
               verbose=True)

# 绘制Pareto前沿
# plot = Scatter(legend=True)
# plot.add(res.F, label="NSGA-II Results", color="red")  # 优化得到的解集
# plot.add(pareto_front, label="True Pareto Front", color="blue")  # 真实Pareto前沿
# plot.show()

# 绘制Pareto前沿
plot = Scatter(legend=True)
plot.add(pareto_front, color="blue", alpha=0.2, s=50, label="True Pareto Front")
plot.add(res.F, color="red", s=10, label="NSGA-II Results") # 真实Pareto前沿
plot.show()

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
plot.show()
