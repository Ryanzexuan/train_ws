import matplotlib.pyplot as plt

# 创建示例数据
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 3, 5, 7, 9]

# 绘制两个轨迹并为它们添加标签
plt.plot(x, y1, label='Line 1')
plt.plot(x, y2, label='Line 2')

# 添加图例（标签）
plt.legend()

# 设置图形的标题和轴标签
plt.title('Trajectory Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图形
plt.show()
