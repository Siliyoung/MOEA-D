from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
import os
from problems import *


def save_image(pareto_front, res,path, file_name,labels):
    # 可视化：绘制优化解
    plot = Scatter(legend=True, labels=labels)
    plot.add(pareto_front, color="blue", alpha=0.2, s=50, label="True Pareto Front")
    plot.add(res.F, color="red", s=10, label="SPEA2 Results")
    # plot.show()
    # 保存图像：使用算法名和问题名命名文件
    image_filename = os.path.join(path, file_name)
    plot.save(image_filename)  # 将图像保存到文件

# 保存结果到文件
def save_results_to_file(algorithm_name, problem_name, generations, igd, gd, hv, filename):
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

    

def handle_zdt(problem_name, n_gen,**kwargs):
    zdt = ZDT()
    problem = zdt.get_problem(problem_name)   # 可换为 "zdt2", "zdt3", 等

    # 定义SPEA2算法
    algorithm = SPEA2(
        pop_size=100,
        eliminate_duplicates=True
    )
    # 执行优化
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        seed=1,
        save_history=False,
        verbose=False
    )

    # 获取zdt问题的真实Pareto前沿
    pareto_front = problem.pareto_front()
    save_image(pareto_front, res,"integaration/images",f"{algorithm.__class__.__name__}_{problem_name}.png",labels=["F1", "F2"])

    # 计算性能指标
    igd = IGD(pf=pareto_front)  # 传递真实Pareto前沿
    gd = GD(pf=pareto_front)  # 计算 GD
    hv = HV(ref_point=[1.1]*problem.n_obj)  # 计算 HV，参考点需要指定

    igd_value = igd.do(res.F)  # 计算 IGD
    gd_value = gd.do(res.F)    # 计算 GD
    hv_value = hv.do(res.F)    # 计算 HV
    # 保存训练结果到文件
    save_result_file = "integaration/results/algorithm_results.csv"
    save_results_to_file(
        algorithm_name="SPEA2",
        problem_name=problem_name,
        generations=n_gen,  # 训练轮次（代数）
        igd=igd_value,    # 计算的 IGD 值
        gd=gd_value,      # 计算的 GD 值
        hv=hv_value,       # 计算的 HV 值
        filename=save_result_file
    )

def handle_dtlz(problem_name, n_gen,**kwargs):
    dtlz = DTLZ()
    problem = dtlz.get_problem(problem_name=problem_name,  n_obj=3)  # 这里传递 `n_var` 和 `n_obj` 会被自动处理
    # problem = zdt.get_problem(problem_name)   # 可换为 "zdt2", "zdt3", 等

    # 定义SPEA2算法
    algorithm = SPEA2(
        pop_size=100,
        eliminate_duplicates=True
    )
    # 执行优化
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        seed=1,
        save_history=False,
        verbose=False
    )

    # 获取zdt问题的真实Pareto前沿
    pareto_front = problem.pareto_front()
    save_image(pareto_front, res,"integaration/images",f"{algorithm.__class__.__name__}_{problem_name}.png",labels=["F1","F2","F3"])

    # 计算性能指标
    igd = IGD(pf=pareto_front)  # 传递真实Pareto前沿
    gd = GD(pf=pareto_front)  # 计算 GD
    hv = HV(ref_point=[1.1]*problem.n_obj)  # 计算 HV，参考点需要指定

    igd_value = igd.do(res.F)  # 计算 IGD
    gd_value = gd.do(res.F)    # 计算 GD
    hv_value = hv.do(res.F)    # 计算 HV
    # 保存训练结果到文件
    save_result_file = "integaration/results/algorithm_results.csv"
    save_results_to_file(
        algorithm_name="SPEA2",
        problem_name=problem_name,
        generations=n_gen,  # 训练轮次（代数）
        igd=igd_value,    # 计算的 IGD 值
        gd=gd_value,      # 计算的 GD 值
        hv=hv_value,       # 计算的 HV 值
        filename=save_result_file
    )

def handle_wfg(problem_name, n_gen,**kwargs):
    wfg = WFG()
    n_obj = 3
    n_var = 12
    k = 4
    problem = wfg.get_problem(problem_name=problem_name,  n_obj=n_obj,n_var=n_var)  # 这里传递 `n_var` 和 `n_obj` 会被自动处理
    # 定义参考方向
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)
    # 定义SPEA2算法
    algorithm = SPEA2(
        pop_size=100,
        eliminate_duplicates=True
    )
    # 执行优化
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        seed=1,
        save_history=False,
        verbose=False
    )

    # 获取zdt问题的真实Pareto前沿
    pareto_front = problem.pareto_front()
    save_image(pareto_front, res,"integaration/images",f"{algorithm.__class__.__name__}_{problem_name}.png",labels=["F1","F2","F3"])

    # 计算性能指标
    igd = IGD(pf=pareto_front)  # 传递真实Pareto前沿
    gd = GD(pf=pareto_front)  # 计算 GD
    hv = HV(ref_point=[1.1]*problem.n_obj)  # 计算 HV，参考点需要指定

    igd_value = igd.do(res.F)  # 计算 IGD
    gd_value = gd.do(res.F)    # 计算 GD
    hv_value = hv.do(res.F)    # 计算 HV
    # 保存训练结果到文件
    save_result_file = "integaration/results/algorithm_results.csv"
    save_results_to_file(
        algorithm_name="SPEA2",
        problem_name=problem_name,
        generations=n_gen,  # 训练轮次（代数）
        igd=igd_value,    # 计算的 IGD 值
        gd=gd_value,      # 计算的 GD 值
        hv=hv_value,       # 计算的 HV 值
        filename=save_result_file
    )

# 使用示例
if __name__ == "__main__":
    handle_zdt("zdt1", 500)
    handle_dtlz("dtlz1", 500)
    handle_wfg("wfg1", 500)
    

