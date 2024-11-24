import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from mpl_toolkits.mplot3d import Axes3D  # 仅在需要时用于3D绘图

# 定义DTLZ系列问题
problems = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]

# 创建子图，考虑到一些问题目标数为3
fig, axes = plt.subplots(3, 3, figsize=(18, 12))

# 遍历每个问题，绘制Pareto前沿
for i, problem_name in enumerate(problems):
    # 获取问题实例，目标数根据问题进行自动选择
    problem = get_problem(problem_name)  
    pareto_front = problem.pareto_front()  # 获取默认的Pareto前沿

    # 获取当前子图
    ax = axes[i//3, i%3]

    # 根据目标数调整绘制方式
    if problem.n_obj == 2:
        # 对于目标数为2，使用二维坐标
        ax.plot(pareto_front[:, 0], pareto_front[:, 1], color='blue', label=f"Pareto Front - {problem_name}")
        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        ax.set_title(f"{problem_name}")
        ax.grid(True)
    elif problem.n_obj == 3:
        # 对于目标数为3，使用3D坐标
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], color='blue', label=f"Pareto Front - {problem_name}")
        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        ax.set_zlabel("F3")
        ax.set_title(f"{problem_name}")
        ax.grid(True)

# 调整子图间距
plt.tight_layout()
plt.show()
