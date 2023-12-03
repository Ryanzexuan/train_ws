import numpy as np

# 创建一个示例的二维数组
arr = np.array([[90, 85, 75],
                [110, 105, 95],
                [80, 75, 70],
                [120, 115, 105],
                [95, 90, 80],
                [130, 125, 115],
                [70, 65, 60],
                [140, 135, 125]])

# 找到第二列中大于100的行的索引
indices = np.where(arr[:, 1] > 100)

# 删除包含大于100的行
arr = np.delete(arr, indices, axis=0)

# 打印结果
print(arr)
