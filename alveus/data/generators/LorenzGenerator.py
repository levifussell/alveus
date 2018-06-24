import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

import datetime

# Global variables
# speed
move_speed = 0.008

# mackey glass params
sigma = 10.0
phi = 28.0
beta = 8./3.


def runLorenz(num_data_samples=5000):
    # ------ better version on wikipedia which just solves the ODE----
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0

    def f(state, t):
        x, y, z = state  # unpack the state vector
        # derivatives
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    delta_t = 0.01
    t_end = num_data_samples * delta_t
    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, t_end, delta_t)

    states = odeint(f, state0, t)

    return states


def onExit(data, axis1=0, axis2=2):
    # save the data
    np.savetxt("data_{}.txt".format(datetime.date.today()), data,
               delimiter=",")

    # plot the data
    plt.plot(data[:, axis1], data[:, axis2], linewidth=0.4)
    plt.xlabel("{}".format(axis1))
    plt.ylabel("{}".format(axis2))
    plt.show()


def d_x(x_t, y_t, z_t):
    return sigma * (y_t - x_t)


def d_y(x_t, y_t, z_t):
    return x_t * (phi - z_t) - y_t


def d_z(x_t, y_t, z_t):
    return x_t * y_t - beta * z_t


if __name__ == "__main__":
    data = runLorenz(num_data_samples=20000)
    onExit(data)
