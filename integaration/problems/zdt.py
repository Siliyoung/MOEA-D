from pymoo.problems import get_problem
import os
import sys

# 将根目录添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ZDT:
    def __init__(self):
        # ZDT问题名与类的映射
        self.problem_dict = {
            "zdt1": "zdt1",
            "zdt2": "zdt2",
            "zdt3": "zdt3",
            "zdt4": "zdt4",
            "zdt5": "zdt5",
            "zdt6": "zdt6"
        }

    def get_problem(self, problem_name, **kwargs):
        """
        根据输入的名称返回相应的 ZDT 问题对象。
        
        :param problem_name: ZDT问题的名称（如 'zdt1', 'zdt2' 等）
        :param kwargs: 其他参数（如目标数、决策变量数等）
        :return: pymoo 问题对象
        """
        if problem_name not in self.problem_dict:
            raise ValueError(f"Unknown problem name: {problem_name}. Supported names are: {', '.join(self.problem_dict.keys())}")
        
        # 传递给 `get_problem` 的参数已包含在 `kwargs` 中
        return get_problem(self.problem_dict[problem_name], **kwargs)

# 使用示例
if __name__ == "__main__":
    # 创建 ZDT 类实例
    zdt = ZDT()
    
    # 获取 ZDT1 问题，假设有 30 个决策变量，目标数为 2
    problem = zdt.get_problem("zdt1")  # 这里传递 `n_var` 和 `n_obj` 会被自动处理

    # 输出问题的一些基本信息
    print(f"Problem: {problem.name}")
    print(f"Number of variables: {problem.n_var}")
    print(f"Number of objectives: {problem.n_obj}")
