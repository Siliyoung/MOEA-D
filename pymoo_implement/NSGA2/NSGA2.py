from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover  # 连续问题的交叉算子
from pymoo.operators.mutation.pm import PolynomialMutation  # 连续问题的变异算子
from pymoo.operators.sampling.rnd import FloatRandomSampling  # 连续问题的采样算子
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 获取ZDT5问题
problem = get_problem("zdt5")

# 定义NSGA-II算法
algorithm = NSGA2(pop_size=100,
                  sampling=FloatRandomSampling(),  # 使用适用于连续优化问题的采样算子
                  crossover=SimulatedBinaryCrossover(),  # 使用适用于连续优化问题的交叉算子
                  mutation=PolynomialMutation(),  # 使用适用于连续优化问题的变异算子
                  eliminate_duplicates=True)

# 执行优化
res = minimize(problem,
               algorithm,
               ('n_gen', 500),
               seed=1,
               verbose=False)


# 获取问题的真实Pareto前沿
pareto_front = problem.pareto_front()

# 输出结果并绘制三维散点图
plot = Scatter(legend=True, labels=["F1", "F2"])
# 可视化：绘制优化解
plot.add(pareto_front, color="blue", alpha=0.2, s=50, label="True Pareto Front")
plot.add(res.F, color="red", s=10, label="NSGA2 Results")
plot.show()


