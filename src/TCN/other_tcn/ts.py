import random
import numpy as np
 
 
def generate_time_series(len):
    backshift = 10
    # np.random.random(size=None)
    # Return random floats in the half-open interval [0.0, 1.0).
    r1 = np.random.random(len + backshift)
    r2 = np.random.random(len + backshift)
 
    # random.choices(population,weights=None,*,cum_weights=None,k=1)
    # 从population中随机选取k次数据，返回一个列表，可以设置权重。 
    # 注意：每次选取都不会影响原序列，每一次选取都是基于原序列。
    # 参数weights设置相对权重，它的值是一个列表，设置之后，每一个成员被抽取到的概率就被确定了。
    # 比如weights=[1,2,3,4,5],那么第一个成员的概率就是P=1/(1+2+3+4+5)=1/15。
    # cum_weights设置累加权重，Python会自动把相对权重转换为累加权重，即如果你直接给出累加权重，
    # 那么就不需要给出相对权重，且Python省略了一步执行。
    # 比如weights=[1,2,3,4],那么cum_weights=[1,3,6,10]
    # 这也就不难理解为什么cum_weights=[1,1,1,1,1]输出全是第一个成员1了。
    rm = [random.choices([0, 0, 0, 1])[0]
          for _ in range(len + backshift)]
 
    ts = np.zeros([len + backshift, 4])
    for i in range(backshift, len + backshift):
        ts[i, 1] = r1[i]
        ts[i, 2] = r2[i]
        ts[i, 3] = rm[i]
 
        ts[i, 0] = ts[i - 1, 0] -\
                   (r1[i - 1] + r1[i - 2]) +\
                   4 * r2[i - 3] * (rm[i - 4] + rm[i - 6])
 
    return ts[backshift:]