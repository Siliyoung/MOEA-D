from pymoo.problems import get_problem
from pymoo.util.reference_directions import get_reference_directions

problem = get_problem("zdt1")
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=99)
print("Problem and reference directions initialized successfully.")