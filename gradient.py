import numpy as np
import matplotlib.pyplot as plt
from math import dist

# Global Variables
obstacle = np.array([0,1])
obstacle_potential = 0.1
goal_potential = 10
goal = np.array([-1, 1])

# Gradient Descent Parameters
start_point = np.array([1, 1])
learning_rate = 5E-4
n_iterations = 500
# Define the function and its gradient
def f(X, Y):    
    return ((1/((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2)) * obstacle_potential + ((X - goal[0]) ** 2 + (Y - goal[1]) ** 2) * goal_potential )


def gradient_f(X, Y):
    try:
        Fxg = -goal_potential*(X - goal[0])/((((X - goal[0]) ** 2 + (Y - goal[1]) ** 2))**0.5)

        Fyg = -goal_potential*(Y - goal[1])/((((X - goal[0]) ** 2 + (Y - goal[1]) ** 2))**0.5)
        Fxo = -obstacle_potential*(X - obstacle[0])/((((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2))**1.5)
        Fyo = -obstacle_potential*(Y - obstacle[1])/((((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2))**1.5)
    except RuntimeWarning:
        Fxg = 0
        Fyg = 0
    Fx = Fxg + Fxo
    Fy = Fyg + Fyo

    # plt.figure()
    # plt.quiver(X, Y, Fx, Fy)
    # plt.show()

    return np.array([Fx, Fy])


# Gradient descent algorithm
def gradient_descent(gradient, start, learning_rate, n_iterations):
    trajectory = [start]
    current = start
    for i in range(n_iterations):
        grad = gradient(*current)
        current = current  + learning_rate * grad
        trajectory.append(current)
        if dist(current, goal) < 0.01:
            print("Path Planning Success")
            print(i)
            break            
    
    print(dist(current, goal))
    return np.array(trajectory)

# Set up the contour plot
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Perform gradient descent

trajectory = gradient_descent(gradient_f, start_point, learning_rate, n_iterations)

# Plot the contour and the trajectory
plt.contour(X, Y, Z, levels=100)
plt.plot(trajectory[:,0], trajectory[:,1], 'ro-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent on Contour Plot')
plt.show()


