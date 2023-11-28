import numpy as np
import matplotlib.pyplot as plt

def f(X,Y):
    # print(np.shape(X))
    obstacle = [0, 0]
    obstacle_potential = 0.1
    goal_potential = 1
    goal_list = [-2, -2]
    # pot = ((1/((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2)) * obstacle_potential + ((X - goal_list[0]) ** 2 + (Y - goal_list[1]) ** 2) * goal_potential )
    try:
        Fxg = -goal_potential*(X - goal_list[0])/((((X - goal_list[0]) ** 2 + (Y - goal_list[1]) ** 2))**0.5)

        Fyg = -goal_potential*(X - goal_list[1])/((((X - goal_list[0]) ** 2 + (Y - goal_list[1]) ** 2))**0.5)
        Fxo = -obstacle_potential*(X - obstacle[0])/((((X[0] - obstacle[0]) ** 2 + (X[1] - obstacle[1]) ** 2))**1.5)
        Fyo = -obstacle_potential*(X[1] - obstacle[1])/((((X[0] - obstacle[0]) ** 2 + (X[1] - obstacle[1]) ** 2))**1.5)
    except:
        Fxg = 0
        Fyg = 0
    Fx = Fxg + Fxo
    Fy = Fyg + Fyo

    return Fx, Fy

def potential(X, Y):
    obstacle = [0,0]
    obstacle_potential = 1
    goal_potential = 100
    goal_list = [-2, -2]
    return ((1/((X - obstacle[0]) ** 2 + (Y - obstacle[1]) ** 2)) * obstacle_potential + ((X - goal_list[0]) ** 2 + (Y - goal_list[1]) ** 2) * goal_potential )

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
# Meshgrid
# X = np.linspace(-5,5)
# Y = np.linspace(-5,5)

x, y= np.meshgrid(np.linspace(-2,2, 100), np.linspace(-2,2, 100))

u, v= f(x,y)

potential_field = potential(x, y)

plot_potential(x, y, potential_field)

print(np.shape(u))

skip = (slice(None, 20), slice(None, 20))

# plt.quiver(x[skip], y[skip], u[skip], v[skip])
plt.quiver(x, y, u, v)

# plt.set(aspect=1, title='Quiver Plot')
plt.show()