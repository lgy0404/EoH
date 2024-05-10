# 测试文章中生成的代码是否可行
import numpy as np

import numpy as np

def heuristic(current_sequence, time_matrix, m, n):
    machine_subset = np.random.choice(m, max(1, int(0.3*m)), replace=False)
    # randomly select a subset of machines
    weighted_avg_execution_time = np.average(time_matrix[:, machine_subset], axis=1,
    weights=np.random.rand(len(machine_subset)))
    # compute the weighted average execution time
    perturb_jobs = np.argsort(weighted_avg_execution_time)[-int(0.3*n):]
    # sort the last jobs based on the weighted average execution time
    new_matrix = time_matrix.copy()
    perturbation_factors = np.random.uniform(0.8, 1.2, size=(len(perturb_jobs), len(
    machine_subset)))
    # calculate perturbation factors, introduce certain randomness
    
    # Fix: Ensure the data type compatibility
    new_matrix = new_matrix.astype(float)  # Convert time_matrix to float type
    new_matrix[perturb_jobs[:, np.newaxis], machine_subset] *= perturbation_factors.astype(float)
    # calculate the final guiding matrix
    return new_matrix, perturb_jobs

# def heuristic(current_sequence, time_matrix, m, n):
#     machine_subset = np.random.choice(m, max(1, int(0.3*m)), replace=False)
#     # randomly select a subset of machines
#     weighted_avg_execution_time = np.average(time_matrix[:, machine_subset], axis=1,
#     weights=np.random.rand(len(machine_subset)))
#     # compute the weighted average execution time
#     perturb_jobs = np.argsort(weighted_avg_execution_time)[-int(0.3*n):]
#     # sort the last jobs based on the weighted average execution time
#     new_matrix = time_matrix.copy()
#     perturbation_factors = np.random.uniform(0.8, 1.2, size=(len(perturb_jobs), len(machine_subset)))
#     # calculate perturbation factors, introduce certain randomness
#     new_matrix[perturb_jobs[:, np.newaxis], machine_subset] *= perturbation_factors
#     # calculate the final guiding matrix
#     return new_matrix, perturb_jobs

# 定义时间矩阵和作业顺序
time_matrix = np.random.randint(1, 10, size=(5, 5))  # 5x5的随机时间矩阵
current_sequence = [0, 1, 2, 3, 4]  # 作业顺序
m = 5  # 机器数量
n = 5  # 作业数量

# 调用heuristic函数
new_matrix, perturb_jobs = heuristic(current_sequence, time_matrix, m, n)


# 打印结果
print("经过扰动后的时间矩阵：")
print(new_matrix)
print("进行扰动的作业索引：", perturb_jobs)
