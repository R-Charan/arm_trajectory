import numpy as np
import matplotlib.pyplot as plt
from math import dist

# Global Variables
start_point = np.array([2, 1])
goal = np.array([-2, 1])
obstacle = np.array([0,1])

# Potential parameters
obstacle_potential = 0.1
goal_potential = 10

# Gradient Descent Parameters
learning_rate = 5E-4

# Define the potential function
def calculate_potential(X, Y):    
    return ((1/((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2)) * obstacle_potential + ((X - goal[0]) ** 2 + (Y - goal[1]) ** 2) * goal_potential )

# Defining the function to calculate the gradient
def gradient_f(X, Y):
    try:
        Fxg = -goal_potential*(X - goal[0])/((((X - goal[0]) ** 2 + (Y - goal[1]) ** 2))**0.5)

        Fyg = -goal_potential*(Y - goal[1])/((((X - goal[0]) ** 2 + (Y - goal[1]) ** 2))**0.5)
        Fxo = obstacle_potential*(X - obstacle[0])/((((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2))**1.5)
        Fyo = obstacle_potential*(Y - obstacle[1])/((((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2))**1.5)
    except RuntimeWarning:
        Fxg = 0
        Fyg = 0
    Fx = Fxg + Fxo
    Fy = Fyg + Fyo

    return np.array([Fx, Fy])


# Gradient descent algorithm
def gradient_descent(gradient, start, learning_rate):
    trajectory = [start]
    current = start
    while(dist(current, goal) > 0.01):
        grad = gradient(*current)
        current = current + learning_rate * grad
        trajectory.append(current)
        if dist(current, goal) < 0.01:
            print("Path Planning Success")
            break            
    
    print(dist(current, goal))
    return np.array(trajectory)

# Plot the potential field on a 3D subplot
def plot_potential(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis (Constant)')
    ax.set_title('3D Potential Field')
    plt.show()
    return

# Set up the Workspace
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Calculating the potential
Z = calculate_potential(X, Y)

# Plot the contour
plot_potential(X, Y, Z)

# Perform gradient descent to determine the trajectory
trajectory = gradient_descent(gradient_f, start_point, learning_rate)

# Plot the contour and the trajectory
fig = plt.figure()
cf = plt.contourf(X, Y, Z, levels=100)
plt.plot(trajectory[:,0], trajectory[:,1], 'ro-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent on Contour Plot')
fig.colorbar(cf)
plt.show()

'''
Note: The logic fails or takes infinite time if the obstacle is exactly in-between the goal and the start point.
One solution is to change the goal by a minimal value like 0.01.
'''

