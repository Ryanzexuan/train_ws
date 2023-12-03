import time
import rospy
import casadi as cs
import math
import tf.transformations as tf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tcn_common import TCNModel
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import ml_casadi.torch as mc
from common import DoubleIntegratorWithLearnedDynamics,MPC

cur_rec_state_set = np.zeros(7)
cur_cmd = np.zeros(2)


def run():
    N = 10  # YZX notes:true predict horizon 
    Nsim = 100
    
    # Get git commit hash for saving the model
    saved_dict = torch.load("/home/ryan/raigor/train_ws/results/model_fitting/TCN/tcn.pt")
    input_size = saved_dict['input_size']
    print(f"input size:{input_size}")
    learned_dyn_model = TCNModel(saved_dict['input_size'], saved_dict['output_size'], 
                                 saved_dict['hidden_size'], saved_dict['hidden_layers'])

    learn_model = learned_dyn_model
    # simulation data read
    rec_dataim = "/home/ryan/raigor/train_ws/data/simplified_sim_dataset/train/dataset_4res.csv"
    raw_data_sim = pd.read_csv(rec_dataim)
    data_pose_sim = np.column_stack((raw_data_sim['x_position_input'], raw_data_sim['y_position_input'], raw_data_sim['yaw_input']))[:300]
    data_ref_sim = np.column_stack((raw_data_sim['x_ref'], raw_data_sim['y_ref'], raw_data_sim['yaw_ref']))[:300]
    print(f"data_pose_sim:\n{data_pose_sim}")
    print(f"data_sim shape:{data_pose_sim.shape[0]}")

    model = DoubleIntegratorWithLearnedDynamics(learn_model)
    print("model init successfully")
    solver = MPC(model=model.model(), N=N).solver
    integrator = MPC(model=model.model(), N=N).simulator

    print('Warming up model...')
    x_l = []
    u_l = []
    input_l = []
    for i in range(N):
        x_l.append(solver.get(i, "x"))
        u_l.append(solver.get(i, "u"))
        input_l = np.hstack((x_l, u_l))

    print('Warmed up!')

    x = []
    y_ref = []
    ts = 1. / N
    xt = np.array([0., 0., 0]) # boundary for x 
    opt_times = []
    # solver.print_statistics()

    simX = []
    simU = []
    simX_b4 = []
    x_current = np.array([cur_rec_state_set[0], cur_rec_state_set[1], cur_rec_state_set[2]])
    simX.append(xt)
    x_draw = []
    y_ref_draw = []
    
    
    ## yzx init yref
    rec_file_1 = "/home/ryan/raigor/train_ws/data/simplified_sim_dataset/train/dataset_addref.csv"
    df_val_pre = pd.read_csv(rec_file_1,index_col=False)
    x_input = np.array(df_val_pre['x_position_input'])
    y_input = np.array(df_val_pre['y_position_input'])
    yaw_input = np.array(df_val_pre['yaw_input'])

    yref = np.column_stack([x_input,y_input, yaw_input])
    x_ref = []
    for ref in enumerate(yref):
        x_ref.append(ref)
    
    pred_h = 3
    Nsim = 0
    i = 0
    goal = False
    control = Twist()
    shape = (0, pred_h, input_size)
    past_state = np.zeros(shape)
    # print(x_ref)
    while(goal is False or rospy.is_shutdown()):
        # solve ocp
        now = time.time()
        # find near ref path
        # x_ref = data_ref_sim
        index = findnearestIndex(x_current, x_ref)
        print(index)

        # set y_ref
        y_cur_ref = Set_y_ref(x_ref, index, N)
        if(y_cur_ref == -1):
            print("find goal!!!")
            goal = True
            break
        y_ref_draw.append(x_ref[index])
        simX_b4.append(x_current)
        y_cur_ref = np.array(y_cur_ref)
        print(f"len(y_cur_ref):{len(y_cur_ref)}")
        y_e = y_cur_ref[len(y_cur_ref)-1]
        solver.set(N, 'yref', y_cur_ref[len(y_cur_ref)-1])
        for j in range(len(y_cur_ref)-1):
            xs_between = np.concatenate((y_cur_ref[j], np.zeros(2)))
            solver.set(j, "yref", xs_between)
        if(N-len(y_cur_ref) > 0):
            for k in range(N-len(y_cur_ref)):
                solver.set(len(y_cur_ref)+k, "yref", np.concatenate((y_e, np.zeros(2))))
        
        ##  set inertial (stage 0)
        # x_current = data_pose_sim[i] # let real sim data be the pose
        solver.set(0, 'lbx', x_current)
        solver.set(0, 'ubx', x_current)
        status = solver.solve()
        solver.get_residuals()
        solver.print_statistics()
        # print(solver.get_residuals())
        if status != 0 :
            continue
            # raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
        
        # print(solver.get(0, 'u'))
        cur_u = np.array(solver.get(0, 'u'))
        control.linear.x = cur_u[0]
        control.angular.z = cur_u[1]
        nn_msg.publish(control)
        # rate.sleep()
        simU.append(solver.get(0, 'u'))
        # simulate system
        integrator.set('x', x_current)
        integrator.set('u', simU[i])

        status_s = integrator.solve()
        if status_s != 0:
            continue
            # raise Exception('acados integrator returned status {}. Exiting.'.format(status))

        # update
        
        x_current = np.array([cur_rec_state_set[0], cur_rec_state_set[1], cur_rec_state_set[2]])
        rospy.loginfo(f"x_current:{x_current}\n")
        simX.append(x_current)        
        x_draw.append(x_current)

        # update NN params into ocp solver params
        # update MLP part
        
        params = learned_dyn_model.approx_params(past_state, flat=True)
        for i in range(N):
            solver.set(i, "p", params[i])

        
        # calc end time for each epoch
        elapsed = time.time() - now
        opt_times.append(elapsed)
        Nsim = Nsim+1
        i = Nsim
        if i==2500:
            break
        print(f"Nsim:{Nsim}")
        for idx in range(past_state.shape[1]-1):
            past_state[0,idx,:] = past_state[0,idx+1,:]
        past_state[0,range(past_state.shape[1]-1)] = x_current


    print(solver.get_stats('residuals'))
    # print(f"x_current:{x_current}")
    # print(f"ref:{x_ref}")
    # plt.figure()
    # plt.grid(True)
    simX = np.array(simX)
    simU = np.array(simU)
    simX_b4 = np.array(simX_b4)
    y_ref_draw = np.array(y_ref_draw)
    # plt.show()
    # Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=xt, target_state=x_ref[len(x_ref)-1], robot_states=simX, )
    # draw(x_draw,'pose with mlp')
    # draw(y_ref_draw,'y ref')
    # draw(data_ref_sim, 'sim y ref')
    # plt.show()
    
    save_data2csv(simX[1:],simU,simX_b4,y_ref_draw)  
    print(f'Mean iteration time with CNN ResNet Model: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')

def draw(data,label):
    x = []
    y = []
    for i in range(len(data)):
        p = data[i]
        x.append(p[0])
        y.append(p[1])
    
    plt.plot(x, y, marker='o', linestyle='-',label=label)
    plt.legend()

    
        
def Set_y_ref(ref_path, idx, Horizon):
    cur_ref_path = []
    min_idx = min(idx + Horizon, len(ref_path)-1)
    if(idx == len(ref_path)-1): 
        return -1
    # print(min_idx)
    print(f"len(ref_path):{len(ref_path)}")
    print(f"idx+H:{idx + Horizon}")
    for i in range(idx, min_idx+1):
        cur_ref_path.append(ref_path[i])
    # print(len(cur_ref_path))
    return cur_ref_path


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

def Callback_base(msg):
    rospy.loginfo("msg got~!!!!!")
    quaternion = msg.pose.pose.orientation
    roll, pitch, yaw = tf.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    # rospy.loginfo(f"x pose{msg.pose.pose.position.x}")
    cur_rec_state_set[0] = msg.pose.pose.position.x 
    cur_rec_state_set[1] = msg.pose.pose.position.y 
    cur_rec_state_set[2] = yaw
    cur_rec_state_set[3] = rospy.Time.now().to_sec()
    cur_rec_state_set[4] = msg.twist.twist.linear.x 
    cur_rec_state_set[5] = msg.twist.twist.linear.y 
    cur_rec_state_set[6] = msg.twist.twist.angular.z

def save_data2csv(next, u, pst, ref):
    # print(f"next:{next}")
    # print(f"u:{u}")
    # print(f"pst:{pst}")
    # print(f"next:{next.shape[0]}")
    # print(f"u:{u.shape[0]}")
    # print(f"pst:{pst.shape[0]}")
    data = pd.DataFrame({'x_position_input': pst[:, 0],
                        'y_position_input': pst[:, 1],
                        'yaw_input': pst[:, 2],
                        'con_x_input': u[:, 0],
                        'con_z_input':u[:, 1],
                        'x_position_output': next[:,0],
                        'y_position_output': next[:,1],
                        'yaw_output': next[:,2],
                        'x_ref': ref[:, 0],
                        'y_ref': ref[:, 1],
                        'yaw_ref': ref[:, 2]})
    data.to_csv("/home/ryan/raigor/train_ws/data/simplified_sim_dataset/train/dataset_gzsim_mlp.csv")

if __name__ == '__main__':
    rospy.init_node("acados", anonymous=True)
    rospy.Subscriber("/base_pose_ground_truth", Odometry, Callback_base)
    nn_msg = rospy.Publisher('/cmd_vel', Twist, queue_size=10)  
    rate = rospy.Rate(1)   
    run()