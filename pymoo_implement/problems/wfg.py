import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
import numpy as np

# 定义WFG系列问题
# problems = ["wfg1", "wfg2", "wfg3"]
# problems = ["wfg4", "wfg5", "wfg6"]
problems = ["wfg7", "wfg8", "wfg9"]
n_var = 12  # WFG问题的变量数
k = 4  # WFG问题的 k 参数

# 创建一个大的画布，2行3列的布局
fig, axs = plt.subplots(3, 2, figsize=(12, 18))  # 3行2列的布局

# 遍历WFG系列问题
for i, problem_name in enumerate(problems):
    # 获取WFG问题的2目标和3目标
    for j, n_obj in enumerate([2, 3]):  # 2目标和3目标
        # 获取问题
        problem = get_problem(problem_name, n_obj=n_obj, n_var=n_var, k=k)
        
        # 获取真实Pareto前沿
        pareto_front = problem.pareto_front()
        
        # 选择当前的子图
        ax = axs[i, j]  # 第i行第j列的子图
        
        # 绘制目标空间
        if n_obj == 2:
            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], color='blue', alpha=0.5, s=20)
            ax.set_xlabel("F1")
            ax.set_ylabel("F2")
            ax.set_title(f"{problem_name.upper()} - 2 Objectives")
        elif n_obj == 3:
            ax = fig.add_subplot(3, 2, 2*i+2, projection='3d')  # 创建一个三维子图
            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], color='blue', alpha=0.5, s=20)
            ax.set_xlabel("F1")
            ax.set_ylabel("F2")
            ax.set_zlabel("F3")
            ax.set_title(f"{problem_name.upper()} - 3 Objectives")
        
# 调整布局，使得子图之间不会重叠
plt.tight_layout()
plt.show()
