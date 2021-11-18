import numpy as np
grid = [[i,j]for i in range(20) for j in range(10)]
# def dist(a):
#     return a[0] + a[1] == 5
print(np.mean(np.array(grid)[:,0]))
# print(np.where(dist(grid),grid,0))