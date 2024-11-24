from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd import GD
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions

import numpy as np
import matplotlib.pyplot as plt

# 获取ZDT1问题
problem = get_problem("zdt1")

# 参考方向，用于MOEA/D
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

# 设置实验参数
pop_size = 100
n_gen = 250
reference_point = np.array([1.1, 1.1])  # 用于计算Hypervolume的参考点

# 定义算法
algorithms = {
    "MOEA/D": MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
    ),
    "NSGA2": NSGA2(
        pop_size=pop_size
    ),
    "SPEA2": SPEA2(
        pop_size=pop_size
    )
}

# 用于保存结果
results = {}

# 运行算法并计算指标
for name, algorithm in algorithms.items():
    # 执行优化
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        seed=1,
        verbose=False
    )

    # 计算指标
    hv = HV(reference_point=reference_point)
    igd = IGD(pf=problem.pareto_front())
    gd = GD()

    hv_value = hv(res.F)
    igd_value = igd(res.F)
    gd_value = gd(res.F)

    # 保存结果
    results[name] = {
        "pareto_front": res.F,
        "hv": hv_value,
        "igd": igd_value,
        "gd": gd_value,
    }

    # 绘制结果
    Scatter().add(res.F).set_title(f"Pareto Front - {name}").show()

# 打印每个算法的性能指标
for name, result in results.items():
    print(f"\n{name} Performance:")
    print(f"  Hypervolume (HV): {result['hv']}")
    print(f"  Inverted Generational Distance (IGD): {result['igd']}")
    print(f"  Generational Distance (GD): {result['gd']}")

# 绘制所有算法的优化结果
plt.figure(figsize=(10, 8))
for name, result in results.items():
    plt.scatter(result['pareto_front'][:, 0], result['pareto_front'][:, 1], label=name)

plt.title("Pareto Front Comparison (MOEA/D, NSGA2, SPEA2)")
plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.legend()
plt.grid(True)
plt.show()
