import numpy as np
a = [[1,0,1,1],[1,0,0,0]]
b = [np.arange(len(a))]*len(a[0])
b = np.rot90(b,3)
c = [np.arange(len(a[0]))]*len(a)
d = [b,c]
print(np.array(a))
print(b)
print(np.array(c))
print(np.array(d))
print(np.mean(np.multiply(a,d)))
print(np.multiply(a,d))
print(np.mean(np.multiply(a,d)[0]))
print(np.mean(np.multiply(a,d)[1]))