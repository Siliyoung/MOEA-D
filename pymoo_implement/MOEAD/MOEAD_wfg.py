from pymoo.algorithms.moo.moead import MOEAD
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
import os

# 定义问题，选择一个WFG系列问题，例如 WFG1
n_var = 12  # 假设问题有 12 个决策变量
problem_name = "wfg1"
problem = get_problem(problem_name, n_obj=3, n_var=n_var, k=4)  # 目标数和变量数

# 获取WFG问题的真实Pareto前沿
pareto_front = problem.pareto_front(n_pareto_points=500)

# 获取参考方向
ref_dirs = get_reference_directions("das-dennis", n_dim=3, n_points=91)  # 2目标问题，99个参考方向

# 定义MOEA/D算法
algorithm = MOEAD(
    ref_dirs=ref_dirs,
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
    verbose=False
)

# 可视化：绘制真实 Pareto 前沿和优化解
plot = Scatter(legend=True, labels=["F1", "F2","F3"])
plot.add(pareto_front, color="blue", alpha=0.2, s=50, label="True Pareto Front")
plot.add(res.F, color="red", s=10, label="MOEA/D Results")
plot.show()

# 保存图像：使用算法名和问题名命名文件
image_filename = os.path.join("pymoo_implement/images", f"{algorithm.__class__.__name__}_{problem_name}.png")
plot.save(image_filename)  # 将图像保存到文件

# 计算性能指标
igd = IGD(pf=pareto_front)  # 传递真实Pareto前沿
gd = GD(pf=pareto_front)  # 计算 GD
hv = HV(ref_point=[1.1]*problem.n_obj)  # 计算 HV，参考点需要指定

igd_value = igd.do(res.F)  # 计算 IGD
gd_value = gd.do(res.F)    # 计算 GD
hv_value = hv.do(res.F)    # 计算 HV


# 保存结果到文件
def save_results_to_file(algorithm_name, problem_name, generations, igd, gd, hv, filename="pymoo_implement/result/algorithm_results.csv"):
    """
    将算法运行结果保存到文件中
    :param algorithm_name: 算法名称
    :param problem_name: 问题名称
    :param generations: 训练轮次
    :param igd: IGD 值
    :param gd: GD 值
    :param hv: HV 值
    :param filename: 文件名
    """
    # 检查文件是否存在，如果不存在则写入表头
    file_exists = os.path.isfile(filename)
    
    with open(filename, "a") as f:
        # 如果文件不存在，则写入表头
        if not file_exists:
            f.write("Algorithm,Problem,Generations,IGD,GD,HV\n")
        
        # 保存指标结果
        f.write(f"{algorithm_name},{problem_name},{generations},{igd},{gd},{hv}\n")

# 保存训练结果到文件
save_results_to_file(
    algorithm_name="MOEA/D",
    problem_name=problem_name,
    generations=500,  # 训练轮次（代数）
    igd=igd_value,    # 计算的 IGD 值
    gd=gd_value,      # 计算的 GD 值
    hv=hv_value       # 计算的 HV 值
)