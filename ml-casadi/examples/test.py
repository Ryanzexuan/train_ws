import numpy as np
import math
import tf.transformations as tf1
import tf
import matplotlib.pyplot as plt

def run(data):
    x = []
    y = []
    # print(data[50,1])
    for j in range(len(data)):
        for p in data[j]:
            # p = data[j]
            print(p)
            print("ok")
            # x.append(p[0])
            # y.append(p[1])
       
def findnearestIndex(cur_pos,ref_path):
    min_dist = 10000
    index = -1
    for i in range(len(ref_path)):
        ref = ref_path[i]
        dx = cur_pos[0] - ref[0]
        dy = cur_pos[1] - ref[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < min_dist:
            min_dist = dist
            index = i
    return index

def draw():

    # 假设你有时间戳和坐标值的列表
    timestamps = [0, 1, 2, 3, 4]  # 时间戳
    x_values = [0, 1, 2, 3, 4]  # x 坐标值
    y_values = [0, 1, 0, 1, 0]  # y 坐标值

    plt.figure()
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.title('轨迹')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    data_ref = []
    t = np.linspace(1, 100, 10)
    # print(t)
    y_xpos = 10 * np.sin(0.0156*t)
    y_ypos = 10 - 10 * np.cos(0.0156*t)
    y_yaw = 0.0156 * t
    yref = np.array([y_xpos,y_ypos, y_yaw]).T
    for t, ref in enumerate(yref):
        data_ref.append(ref)
        quaternion = tf1.quaternion_from_euler(0, 0, ref[2])
        theta1 = tf1.euler_from_quaternion(quaternion)
        # print(theta1)
        # print(theta1[2] - ref[2])
    draw()



    # print(data_ref)
    # # run(data_ref)
    # index = findnearestIndex([5,5],data_ref)
    # print(index)