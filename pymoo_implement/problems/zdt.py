import matplotlib.pyplot as plt
from pymoo.problems import get_problem

# 定义ZDT系列问题
problems = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt5", "zdt6"]

# 创建子图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 遍历每个问题，绘制Pareto前沿
for i, problem_name in enumerate(problems):
    # 获取问题实例
    problem = get_problem(problem_name)  
    pareto_front = problem.pareto_front()  # 无需传递n_pareto_points

    # 获取当前子图
    ax = axes[i//3, i%3]
    ax.plot(pareto_front[:, 0], pareto_front[:, 1], color='blue', label=f"Pareto Front - {problem_name}")
    ax.set_title(f"{problem_name}")
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    ax.grid(True)

# 调整子图间距
plt.tight_layout()
plt.show()
