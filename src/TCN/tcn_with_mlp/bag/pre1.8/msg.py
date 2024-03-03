# python base
import rospy
import pandas as pd
import numpy as np
import os 
import tf.transformations as tf
import argparse


parser = argparse.ArgumentParser(description='Msg pre processing')
parser.add_argument('--doc', type=str)
input_parser = parser.parse_args()
print(f'args:{input_parser}')
doc = input_parser.doc


def time_preprocess(data):
    print(f'time:{data}')
    time = np.zeros(data.shape)
    time[0] = 0
    init = data[0]
    time[1:] = data[1:] - init
    print(f'time:{time}')
    return time

def quaternion2angle(q):
    angles = []
    for item in q:
        roll, pitch, yaw = tf.euler_from_quaternion(item)
        angles.append(yaw)
    return np.array(angles)

def save_data2csv(time, pose, vel, ctrl, name):
    print(len(time), len(pose), len(vel), len(ctrl))  # 打印数组长度
    # print(f"next:{next.shape[0]}")
    # print(f"u:{u.shape[0]}")
    # print(f"pose:{pose.shape[0]}")
    # print(f"pose:{next_vel.shape[0]}")
    # print(f"pose:{pose_vel.shape[0]}")
    cur_path = os.getcwd()
    csv_name = name
    data = pd.DataFrame({'con_time' : time[:],
                        'x_position_input': pose[:, 0],
                        'y_position_input': pose[:, 1],
                        'z_position_input': pose[:, 2],
                        'x_orientation': pose[:, 3],
                        'y_orientation': pose[:, 4],
                        'z_orientation': pose[:, 5],
                        'w_orientation': pose[:, 6],
                        'yaw_input' : pose[:,7],
                        'vel_linear_x':vel[:,0],
                        'vel_linear_y':vel[:,1],
                        'vel_linear_z':vel[:,2],
                        'vel_angular_x':vel[:,3],
                        'vel_angular_y':vel[:,4],
                        'vel_angular_z':vel[:,5],
                        'con_x_input': ctrl[:, 0],
                        'con_z_input':ctrl[:, 1]})  
    data.to_csv(os.path.join(cur_path, csv_name), index=False)

if __name__ == "__main__":
    input_bag_file = doc
    out_name = doc.replace('_pre', '')
    print(f'doc:{doc}')
    print(f'outname:{out_name}')
    output_bag_file = "big_flat1.csv"
    data = pd.read_csv(input_bag_file)
    # time
    time = data['%time']
    time = time_preprocess(time*10**(-9))
    # ctrl
    con_x = data['field.linear.x']
    con_z = data['field.angular.z']
    ctrl = np.column_stack([con_x, con_z])
    print(f'ctr:{ctrl.shape}')
    # pose 
    position_x = data['field.pose.pose.position.x']	
    position_y = data['field.pose.pose.position.y']	
    position_z = data['field.pose.pose.position.z']	
    orientation_x = data['field.pose.pose.orientation.x']	
    orientation_y = data['field.pose.pose.orientation.y']	
    orientation_z = data['field.pose.pose.orientation.z']	
    orientation_w = data['field.pose.pose.orientation.w']
    quaternion = np.column_stack([orientation_x, orientation_y, orientation_z, orientation_w])
    yaw = quaternion2angle(quaternion)
    pose = np.column_stack([position_x, position_y, position_z, orientation_x, orientation_y, orientation_z, orientation_w, yaw])
    
    # speed 
    vel_x = data['field.twist.twist.linear.x']
    vel_y = data['field.twist.twist.linear.y']					
    vel_z = data['field.twist.twist.linear.z']
    angular_x = data['field.twist.twist.angular.x']
    angular_y = data['field.twist.twist.angular.y']
    angular_z = data['field.twist.twist.angular.z']
    vel = np.column_stack([vel_x, vel_y, vel_z, angular_x, angular_y, angular_z])        
    save_data2csv(time, pose, vel, ctrl, out_name)


