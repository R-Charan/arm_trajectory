import math
import cvxpy
import numpy as np
import matplotlib.pyplot as mp
from scipy.optimize import minimize
import time
from mpl_toolkits.mplot3d import Axes3D
from math import atan2, dist

# Setup params
l1 = 1.0 #link lengh one in m
l2 = 1.0 #link lengh two in m
DT = 0.1
horizon_length = 10
angular_rate = 0.2
# number_of_states = 4
# number_of_control_inputs = 2
# R = np.diag([0.5, 0.5])  # input cost matrix
# Q = np.diag([100.0, 100.0, 0.0, 0.0])  # state cost matrix
# MPC helper
windowSize = 5

# Plotter setup  
mp.close('all')
mp.ion()  
fig, ax = mp.subplots()
mp.axis('equal')
mp.axis([-1.0*windowSize, windowSize, -1.0*windowSize, windowSize])
fig_workspace = mp.Circle((0, 0), 2.0, color='blue', alpha=0.1)
ax.add_patch(fig_workspace)

# Plot goal
fig_goal_handle, = mp.plot(0, 0, 'go', ms=10.0)

# Plot obs
fig_obs_handle, = mp.plot(0, 0, 'ko', ms=10.0)

# Plot robot as such
mp.plot(0, 0, 'ko', ms=4.0)
fig_cur_robot_dot = []
temp, = mp.plot(0, 0, 'ro', ms=4.0)
fig_cur_robot_dot.append(temp)
temp, = mp.plot(0, 0, 'ro', ms=4.0)
fig_cur_robot_dot.append(temp)
fig_cur_robot_line = []
temp, = mp.plot([0, 0], [0, 0], '-r', alpha=1.0)
fig_cur_robot_line.append(temp)
temp, = mp.plot([0, 0], [0, 0], '-r', alpha=1.0)
fig_cur_robot_line.append(temp)

# class Vector2d():
#     """
#     2维向量, 支持加减, 支持常量乘法(右乘)
#     """

#     def __init__(self, x, y):
#         self.deltaX = x
#         self.deltaY = y
#         self.length = -1
#         self.direction = [0, 0]
#         self.vector2d_share()

#     def vector2d_share(self):
#         if type(self.deltaX) == type(list()) and type(self.deltaY) == type(list()):
#             deltaX, deltaY = self.deltaX, self.deltaY
#             self.deltaX = deltaY[0] - deltaX[0]
#             self.deltaY = deltaY[1] - deltaX[1]
#             self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2) * 1.0
#             if self.length > 0:
#                 self.direction = [self.deltaX / self.length, self.deltaY / self.length]
#             else:
#                 self.direction = None
#         else:
#             self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2) * 1.0
#             if self.length > 0:
#                 self.direction = [self.deltaX / self.length, self.deltaY / self.length]
#             else:
#                 self.direction = None

#     def coordinate(self):
#         coordinate = [self.deltaX, self.deltaY]

#         return coordinate

#     def __add__(self, other):
#         """
#         + 重载
#         :param other:
#         :return:
#         """
#         vec = Vector2d(self.deltaX, self.deltaY)
#         vec.deltaX += other.deltaX
#         vec.deltaY += other.deltaY
#         vec.vector2d_share()
#         return vec

#     def __sub__(self, other):
#         vec = Vector2d(self.deltaX, self.deltaY)
#         vec.deltaX -= other.deltaX
#         vec.deltaY -= other.deltaY
#         vec.vector2d_share()
#         return vec

#     def __mul__(self, other):
#         vec = Vector2d(self.deltaX, self.deltaY)
#         vec.deltaX *= other
#         vec.deltaY *= other
#         vec.vector2d_share()
#         return vec

#     def __truediv__(self, other):
#         return self.__mul__(1.0 / other)

#     def __repr__(self):
#         return 'Vector deltaX:{}, deltaY:{}, length:{}, direction:{}'.format(self.deltaX, self.deltaY, self.length,
#                                                                              self.direction)


def plot_obs_and_goal(goal, obs):
    #Plot goal
    fig_goal_handle.set_xdata(goal[0])
    fig_goal_handle.set_ydata(goal[1])

    #Plot obstacle
    fig_obs_handle.set_xdata(obs[0])
    fig_obs_handle.set_ydata(obs[1])

    return


def plot_robo(thetas):
    mp.plot(0, 0, 'ko', ms=4.0)
    #Plot robot as such
    fig_cur_robot_dot[0].set_xdata(l1 * math.cos(thetas[0]))
    fig_cur_robot_dot[0].set_ydata(l1 * math.sin(thetas[0]))
    fig_cur_robot_dot[1].set_xdata(l1 * math.cos(thetas[0]) + l2 * math.cos(thetas[0] + thetas[1]))
    fig_cur_robot_dot[1].set_ydata(l1 * math.sin(thetas[0]) + l2 * math.sin(thetas[0] + thetas[1]))

    fig_cur_robot_line[0].set_xdata([0.0, l1 * math.cos(thetas[0])])
    fig_cur_robot_line[0].set_ydata([0.0, l1 * math.sin(thetas[0])])
    fig_cur_robot_line[1].set_xdata([l1 * math.cos(thetas[0]), l1 * math.cos(thetas[0]) + l2 * math.cos(thetas[0] + thetas[1])])
    fig_cur_robot_line[1].set_ydata([l1 * math.sin(thetas[0]), l1 * math.sin(thetas[0]) + l2 * math.sin(thetas[0] + thetas[1])])
    fig.canvas.draw()
    fig.canvas.flush_events()
    return

def inv_kin(goal):
    xd = goal[0]
    yd = goal[1]

    #solve
    c_theta2 = (((xd**2) + (yd**2)  - (l1**2) - (l2**2)) / (2*l1*l2))
    s_theta2 = math.sqrt(1.0 - (c_theta2**2))
    theta2 = math.atan2(s_theta2,c_theta2)
    M = l1 + l2*math.cos(theta2)
    N = l2*math.sin(theta2)
    gamma = math.atan2(N,M)
    theta1 = math.atan2(yd,xd) - gamma
    return [theta1, theta2]

# Function to generate 3D potential field
def generate_3d_potential_field(workspace_size, obstacle, goal, obstacle_potential, goal_potential):
    x = np.linspace(-workspace_size, workspace_size, 100)
    y = np.linspace(-workspace_size, workspace_size, 100)
    X, Y = np.meshgrid(x, y)
    # Potential field parameters
    

    # Calculate potential field in 2D
    potential_field = ((1/((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2)) * obstacle_potential + ((X - goal[0]) ** 2 + (Y - goal[1]) ** 2) * goal_potential )

    # Extend to 3D by adding a constant Z value
    Z = potential_field # You can adjust the constant Z value as needed

    return X, Y, Z, potential_field

# Visualize the 3D potential field
def plot_3d_potential_field(X, Y, Z, potential_field):
    fig = mp.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, facecolors=mp.cm.viridis(potential_field / np.max(potential_field)), rstride=1, cstride=1, alpha=0.8, antialiased=True)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis (Constant)')
    ax.set_title('3D Potential Field')
    mp.show()

    return 

# def path_planner(x, y, obs_potential, goal_potential, start, goal, obstacle):
#     step_size = 1
#     force_vector = []
#     current_pose = Vector2d(start[0], start[1])
#     goal = Vector2d(goal[0], goal[1])
#     obstacle = Vector2d(obstacle[0], obstacle[1])
#     iter = 1
#     delta_t = 0.01
#     goal_threashold = 0.01
#     max_iters = 500

#     path = []
#     while(iter < max_iters and (current_pose - goal).length > goal_threashold):
#         current_pose_list = current_pose.coordinate()
#         goal_list = goal.coordinate()
#         obstacle_list = obstacle.coordinate()
#         

#         Fyg = -goal_potential*(current_pose_list[1] - goal_list[1])/((((current_pose_list[0] - goal_list[0]) ** 2 + (current_pose_list[1] - goal_list[1]) ** 2))**0.5)

#         Fxo = -obs_potential*(current_pose_list[0] - obstacle_list[0])/((((current_pose_list[0] - obstacle_list[0]) ** 2 + (current_pose_list[1] - obstacle_list[1]) ** 2))**1.5)
#         Fyo = -obs_potential*(current_pose_list[1] - obstacle_list[1])/((((current_pose_list[0] - obstacle_list[0]) ** 2 + (current_pose_list[1] - obstacle_list[1]) ** 2))**1.5)

#         Fx = Fxg + Fxo
#         Fy = Fyg + Fyo

#         iter += 1

        
#         current_pose += (Vector2d(Fx, Fy) * step_size)
#         path.append([current_pose.deltaX, current_pose.deltaY])
#         mp.plot(current_pose.deltaX, current_pose.deltaY, '.b')
#         mp.pause(delta_t)
#     if (current_pose - goal).length <= goal_threashold:
#             print("Path plan success")
#     return

# Function to calculate gradient of the potential field
def gradient_descent(goal, obstacle, start, potential_fields):
    X = start[0]
    Y = start[1]
    num_steps = 1500
    learning_rate = 0.001
    goal_potential = 1
    obs_potential = 1
    traj = []

    for i in range(num_steps):

        gradient_x = -goal_potential*(X - goal[0])/((((X - goal[0]) ** 2 + (Y - goal[1]) ** 2))**0.5) +  obs_potential*(X - obstacle[0])/((((X- obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2))**1.5)
        gradient_y = -goal_potential*(Y - goal[1])/((((X - goal[0]) ** 2 + (Y - goal[1]) ** 2))**0.5) +  obs_potential*(Y - obstacle[1])/((((X- obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2))**1.5)
        X = X - (learning_rate*gradient_x)
        Y = Y + (learning_rate*gradient_y)
        
        if i%50 == 0:
            print(X, Y)

        traj.append([X, Y])
        if dist([X,Y], goal) < 0.05:
            print("planning successful")
            break

    return traj

    

def main():
    #plot setup
    workspace_size = 2
    start = [0, -2]
    goal = [-1,1]
    obstacle = [0, 1.2]
    obstacle_potential = 0.01
    goal_potential = 1
    plot_obs_and_goal(goal, obstacle)

    theta_start = inv_kin(start)
    plot_robo([theta_start[0], theta_start[1]]) 


    #generate potential field
    X, Y, Z, potential_field = generate_3d_potential_field(workspace_size, obstacle, goal, obstacle_potential, goal_potential)
    # print("z", Z )

    # path_planner(X, Y, obstacle_potential, goal_potential, start, goal, obstacle)
    


    trajectory = gradient_descent(goal, obstacle, start, potential_field)

    # reference_trajectory = generate_reference_trajectory(start, gradient_x, gradient_y, goal, num_steps=10)

        # Display the 3D potential field
    plot_3d_potential_field(X, Y, Z, potential_field)

    mp.figure()
    mp.plot(trajectory[0], trajectory[1])
    input(mp.show())

    # print(reference_trajectory)

    print('Program ended')

if __name__ == '__main__':
    main()