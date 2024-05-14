from gurobipy import *
import numpy as np
import time
 
# 参数设定
sequence=[] # 加工的次序
n = 20 # 工件总数
m = 30 # 机器总数
low = 1 # 加工时间最小值
high = 99 # 加工时间最大值
 
# 生成随机数据
np.random.seed(1) # 设置随机种子
 
def produceData(n, m, low, high):
    data = np.zeros([m, n], dtype=int)
    data[:] = np.random.uniform(low, high + 1, [m, n])
    data = data.T
    return data
 
t = produceData(n,m,low,high) # 每个工件在每台机器上的加工时间
 
start_time = time.time() # 记录开始时间
 
# 创建模型
model = Model('FSP')
 
# 创建变量
x = model.addVars(n,n,vtype=GRB.BINARY,name='x')
 
C = model.addVars(n,m,vtype=GRB.CONTINUOUS,name='C')
 
makespan = model.addVar(vtype=GRB.CONTINUOUS,name='makespan')
 
# 设置目标函数
model.addConstrs(makespan >= C[k,m-1] for k in range(n))
 
model.setObjective(makespan, GRB.MINIMIZE)
 
# 约束条件
model.addConstrs(quicksum(x[i,k] for k in range(n))==1 for i in range(n))
 
model.addConstrs(quicksum(x[i,k] for i in range(n))==1 for k in range(n))
 
model.addConstr(C[0,0] >= quicksum(x[i,0] * t[i][0] for i in range(n)))
 
model.addConstrs(C[k+1,j] >= C[k,j] + quicksum(x[i,k+1] * t[i][j] for i in range(n)) for k in range(n-1) for j in range(m))
 
model.addConstrs(C[k,j+1] >= C[k,j] + quicksum(x[i,k] * t[i][j+1] for i in range(n)) for k in range(n) for j in range(m-1))
 
model.addConstrs(C[k,j] >= 0 for k in range(n) for j in range(m))
 
# 设置求解器的最长运行时间为1小时（3600s）
model.setParam(GRB.Param.TimeLimit,3600)
 
# 模型求解
model.optimize()
 
# 记录结束时间
end_time = time.time()
solve_time = end_time - start_time
 
# 打印结果
if model.status == GRB.OPTIMAL:
    print('最优解：')
    for i in range(n):
        for k in range(n):
            if x[i,k].x > 0.5:
                sequence.append(k)
                print(f"{i}是排列Π的第{k}个工件")
    print('工件的加工次序：',sequence)
    print('总完成时间：',model.objVal)
elif model.status == GRB.Status.TIME_LIMIT:
    print('目标函数值上界：',model.objVal,'目标函数值下界：',model.ObjBound)
else:
    print('无可行解')
print("求解时间：",solve_time,'s')