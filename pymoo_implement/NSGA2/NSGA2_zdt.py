from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 获取ZDT1问题
problem = get_problem("zdt1")

# 获取ZDT1问题的真实Pareto前沿
pareto_front = problem.pareto_front()

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


plot = Scatter(legend=True, labels=["F1", "F2"])
# 可视化：绘制优化解
plot.add(pareto_front, color="blue", alpha=0.2, s=50, label="True Pareto Front")
plot.add(res.F, color="red", s=10, label="NSGA-II Results")
plot.show()