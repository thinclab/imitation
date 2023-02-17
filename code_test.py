import numpy as np
# s = []
# for i in range(1000):
#     s.append(np.random.uniform(np.array([-1,-1]),np.array([1,1]),(2,)))

# s = np.array(s)

# print(np.sum(np.sum(s<0, axis=1)==2))
# print(np.sum(s[1]>-1 and s[1]<0))

# a,b = np.array([-1,-1]),np.array([1,1])
# print(np.prod(b-a))

# from scipy.stats import multivariate_normal
# x, y = np.mgrid[-1:1:.01, -1:1:.01]
# pos = np.dstack((x, y))
# rv = multivariate_normal([0.5, -0.2], [[0.1, 0.0], [0.0, 0.1]])
# print(rv.pdf(np.array([0.5, -0.2])))

# sum_size_parts_rd = np.round(0.25200000405311584,3)
sum_size_parts_rd = float(str(0.25200000405311584)[:len(str(0.001))])
print(sum_size_parts_rd)