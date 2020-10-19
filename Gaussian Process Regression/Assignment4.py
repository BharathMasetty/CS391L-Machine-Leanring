import numpy as np
from scipy.optimize import minimize
import random
import sklearn.gaussian_process as gp
from matplotlib import pyplot as plt

filename1 = 'data_GP/data_GP/AG/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213203046-59968-right-speed_0.500.csv'
filename2 = 'data_GP/data_GP/AG/block2-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204004-59968-right-speed_0.500.csv'
filename3 = 'data_GP/data_GP/AG/block3-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204208-59968-right-speed_0.500.csv'
filename4 = 'data_GP/data_GP/AG/block4-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213204925-59968-right-speed_0.500.csv'
filename5 = 'data_GP/data_GP/AG/block5-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161213210121-59968-right-speed_0.500.csv'

data1 = np.genfromtxt(filename1, delimiter=',')
data2 = np.genfromtxt(filename2, delimiter=',')
data3 = np.genfromtxt(filename3, delimiter=',')
data4 = np.genfromtxt(filename4, delimiter=',')
data5 = np.genfromtxt(filename5, delimiter=',')

timesteps = data1[1:, 0]

# Marker zero - use 1 and 4
z_0 = np.empty((5, len(timesteps)))
y_0 = np.empty((5, len(timesteps)))
x_0 = np.empty((5, len(timesteps)))

# Using finger_y
z_0[0, :] = data1[1:, 7]
z_0[1, :] = data2[1:, 7]
z_0[2, :] = data3[1:, 7]
z_0[3, :] = data4[1:, 7]
z_0[4, :] = data5[1:, 7]

y_0[0, :] = data1[1:, 6]
y_0[1, :] = data2[1:, 6]
y_0[2, :] = data3[1:, 6]
y_0[3, :] = data4[1:, 6]
y_0[4, :] = data5[1:, 6]

x_0[0, :] = data1[1:, 5]
x_0[1, :] = data2[1:, 5]
x_0[2, :] = data3[1:, 5]
x_0[3, :] = data4[1:, 5]
x_0[4, :] = data5[1:, 5]

plt.figure(1)
plt.plot()
plt.plot(timesteps, z_0[0, :], color='r', label="0")
plt.plot(timesteps, z_0[1, :], color='c', label="1")
plt.plot(timesteps, z_0[2, :], color='g', label="2")
plt.plot(timesteps, z_0[3, :], color='y', label="3")
plt.plot(timesteps, z_0[4, :], color='b', label="4")
plt.legend()
plt.xlabel("timesteps")
plt.ylabel("motion")
plt.title("Marker finger_z")
plt.savefig("plots/Marker_finger_z.png")

plt.figure(2)
plt.plot()
plt.plot(timesteps, x_0[0, :], color='r', label="0")
plt.plot(timesteps, x_0[1, :], color='c', label="1")
plt.plot(timesteps, x_0[2, :], color='g', label="2")
plt.plot(timesteps, x_0[3, :], color='y', label="3")
plt.plot(timesteps, x_0[4, :], color='b', label="4")
plt.legend()
plt.xlabel("timesteps")
plt.ylabel("motion")
plt.title("Marker finger_x")
plt.savefig("plots/Marker_finger_x.png")


plt.figure(3)
plt.plot()
plt.plot(timesteps, y_0[0, :], color='r', label="0")
plt.plot(timesteps, y_0[1, :], color='c', label="1")
plt.plot(timesteps, y_0[2, :], color='g', label="2")
plt.plot(timesteps, y_0[3, :], color='y', label="3")
plt.plot(timesteps, y_0[4, :], color='b', label="4")
plt.legend()
plt.xlabel("timesteps")
plt.ylabel("motion")
plt.title("Marker finger_y")
plt.savefig("plots/Marker_finger_y.png")


print(timesteps.size)
print(z_0.shape)


# GPR function - returns parameters and y_mean and y_cov.

def GPR(x, y):
    # sampling
    xtr = []
    ytr = []
    length = len(x)

    # choosing one in every 5 points
    for i in range(0, length, 5):
        xtr.append(x[i])
        ytr.append(y[random.choice([0, 1, 2, 3, 4]), i])
    xtr = np.asarray(xtr)
    ytr = np.asarray(ytr)

    kernel = gp.kernels.ConstantKernel(1, (1e-3, 1e+3)) * gp.kernels.RBF(10, (1e-3, 1e3)) \
             + gp.kernels.WhiteKernel(1e-5, (1e-11, 10))
    model = gp.GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=10)
    model.fit(xtr[:, None], ytr)
    params = model.kernel_.get_params()
    parameters = np.asarray(
        [params['k1__k1__constant_value'], params['k1__k2__length_scale'], params['k2__noise_level']])
    y_mean, y_err = model.predict(x[:, None], return_std=True)

    return y_mean, y_err, parameters, xtr, ytr


# Global Fitting
# z direction
Gz_mean, Gz_err, Gz_params, Gz_xtr, Gz_ytr = GPR(timesteps, z_0)
# y direction
Gy_mean, Gy_err, Gy_params, Gy_xtr, Gy_ytr = GPR(timesteps, y_0)
# x direction
Gx_mean, Gx_err, Gx_params, Gx_xtr, Gx_ytr = GPR(timesteps, x_0)

print("Completed Global Fitting")

# Window fitting
window = 100
slide = 10
n_points = len(timesteps)
n_steps = int((n_points - window) / slide + 1)
W_params = np.empty((3, 3))
W_mean = []
W_err = []
w_count = np.zeros(len(timesteps))

for i in range(n_steps):

    time = timesteps[i * slide: i * slide + window]
    wx = x_0[:, i * slide: i * slide + window]
    wy = y_0[:, i * slide: i * slide + window]
    wz = z_0[:, i * slide: i * slide + window]
    x_mean, x_err, x_param, xxtr, xytr = GPR(time, wx)
    y_mean, y_err, y_param, yxtr, yytr = GPR(time, wy)
    z_mean, z_err, z_param, zxtr, zytr = GPR(time, wz)

    if i == 0:
        Wx_mean = x_mean
        Wy_mean = y_mean
        Wz_mean = z_mean

        Wx_err = x_err
        Wy_err = y_err
        Wz_err = z_err

        Wx_params = x_param
        Wy_params = y_param
        Wz_params = z_param

        Wx_xtr = xxtr
        Wy_xtr = xytr
        Wz_xtr = zxtr

        Wx_ytr = xytr
        Wy_ytr = yytr
        Wz_ytr = zytr

    else:
        Wx_mean = np.vstack((Wx_mean, x_mean))
        Wy_mean = np.vstack((Wy_mean, y_mean))
        Wz_mean = np.vstack((Wz_mean, z_mean))

        Wx_err = np.vstack((Wx_err, x_err))
        Wy_err = np.vstack((Wy_err, y_err))
        Wz_err = np.vstack((Wz_err, z_err))

        Wx_params = np.vstack((Wx_params, x_param))
        Wy_params = np.vstack((Wy_params, y_param))
        Wz_params = np.vstack((Wz_params, z_param))

        Wx_xtr = np.append(Wx_xtr, xxtr[17:19])
        Wy_xtr = np.append(Wy_xtr, yxtr[17:19])
        Wz_xtr = np.append(Wz_xtr, zxtr[17:19])

        Wx_ytr = np.append(Wx_ytr, xytr[17:19])
        Wy_ytr = np.append(Wy_ytr, yytr[17:19])
        Wz_ytr = np.append(Wz_ytr, zytr[17:19])

    # Update window count
    w_count[i * slide: i * slide + window] += 1

print("Finished window slide")

# Collecting local params for each point

x_sigy = np.zeros(len(timesteps))
x_sign = np.zeros(len(timesteps))
x_l = np.zeros(len(timesteps))

y_sigy = np.zeros(len(timesteps))
y_sign = np.zeros(len(timesteps))
y_l = np.zeros(len(timesteps))

z_sigy = np.zeros(len(timesteps))
z_sign = np.zeros(len(timesteps))
z_l = np.zeros(len(timesteps))

for i in range(n_steps):
    # x
    x_sigy[i * slide: i * slide + window] += np.full(window, Wx_params[i, 0])
    x_l[i * slide: i * slide + window] += np.full(window, Wx_params[i, 1])
    x_sign[i * slide: i * slide + window] += np.full(window, Wx_params[i, 2])
    # y
    y_sigy[i * slide: i * slide + window] += np.full(window, Wy_params[i, 0])
    y_l[i * slide: i * slide + window] += np.full(window, Wy_params[i, 1])
    y_sign[i * slide: i * slide + window] += np.full(window, Wy_params[i, 2])
    # z
    z_sigy[i * slide: i * slide + window] += np.full(window, Wz_params[i, 0])
    z_l[i * slide: i * slide + window] += np.full(window, Wz_params[i, 1])
    z_sign[i * slide: i * slide + window] += np.full(window, Wz_params[i, 2])

x_sigy = np.divide(x_sigy, w_count)
x_l = np.divide(x_l, w_count)
x_sign = np.divide(x_sign, w_count)

y_sigy = np.divide(y_sigy, w_count)
y_l = np.divide(y_l, w_count)
y_sign = np.divide(y_sign, w_count)

z_sigy = np.divide(z_sigy, w_count)
z_l = np.divide(z_l, w_count)
z_sign = np.divide(z_sign, w_count)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# Awesome PLotes

# for x
plt.figure(4)
plt.plot()
plt.plot(timesteps, x_0[1, :],color='black', label="motion")
plt.plot(timesteps, np.sqrt(x_sigy), color = 'b' ,label = "sigma_y")
plt.plot(timesteps, np.full(len(timesteps), np.sqrt(Gx_params[0])), color=lighten_color('b', 0.2), label='global sigma_y')
plt.plot(timesteps, np.sqrt(x_sign), color = 'g' , label = "sigma_n")
plt.plot(timesteps, np.full(len(timesteps), np.sqrt(Gx_params[2])), color=lighten_color('g', 0.2), label='global sigma_n')
plt.plot(timesteps, np.sqrt(x_l), color = 'r' , label = "l")
plt.plot(timesteps, np.full(len(timesteps), np.sqrt(Gx_params[1])), color=lighten_color('r', 0.2), label='global sigma_l')
plt.legend()
plt.xlabel("timesteps")
plt.title("Parameter variation of marker: finger_x")
plt.yscale("symlog")
plt.savefig("plots/master_x.png")


plt.figure(5)
plt.plot()
plt.plot(timesteps, y_0[1, :],color='black', label="motion")
plt.plot(timesteps, np.sqrt(y_sigy), color = 'b' ,label = "sigma_y")
plt.plot(timesteps, np.full(len(timesteps), np.sqrt(Gy_params[0])), color=lighten_color('b', 0.2), label='global sigma_y')
plt.plot(timesteps, np.sqrt(y_sign), color = 'g' , label = "sigma_n")
plt.plot(timesteps, np.full(len(timesteps), np.sqrt(Gy_params[2])), color=lighten_color('g', 0.2), label='global sigma_n')
plt.plot(timesteps, np.sqrt(y_l), color = 'r' , label = "l")
plt.plot(timesteps, np.full(len(timesteps), np.sqrt(Gy_params[1])), color=lighten_color('r', 0.2), label='global l')
plt.legend()
plt.xlabel("timesteps")
plt.title("Parameter variation of marker: finger_y")
plt.yscale("symlog")
plt.savefig("plots/master_y.png")


plt.figure(6)
plt.plot()
plt.plot(timesteps, z_0[1, :],color='black', label="motion")
plt.plot(timesteps, np.sqrt(z_sigy), color = 'b' ,label = "sigma_y")
plt.plot(timesteps, np.full(len(timesteps), np.sqrt(Gz_params[0])), color=lighten_color('b', 0.2), label='global sigma_y')
plt.plot(timesteps, np.sqrt(z_sign), color = 'g' , label = "sigma_n")
plt.plot(timesteps, np.full(len(timesteps), np.sqrt(Gz_params[2])), color=lighten_color('g', 0.2), label='global sigma_n')
plt.plot(timesteps, np.sqrt(z_l), color = 'r' , label = "l")
plt.plot(timesteps, np.full(len(timesteps), np.sqrt(Gz_params[1])), color=lighten_color('r', 0.2), label='global l')
plt.legend()
plt.xlabel("timesteps")
plt.title("Parameter variation of marker: finger_z")
plt.yscale("symlog")
plt.savefig("plots/master_z.png")


# PLotting GLobal posterior
plt.figure(7)
plt.plot()
plt.plot(timesteps, Gx_mean, '-', color = 'blue', label = "mean")
plt.fill_between(timesteps, Gx_mean - 2* Gx_err, Gx_mean + 2* Gx_err, color='blue', alpha = 0.2, label = "std")
plt.scatter(Gx_xtr, Gx_ytr, s = 5)
plt.title("Globle fitting for finger_x")
plt.xlabel("timesteps")
plt.ylabel("Predicted Value")
plt.legend()
plt.savefig("plots/globlefitting_finger_x.png")

plt.figure(8)
plt.plot()
plt.plot(timesteps, Gy_mean, '-', color = 'red',label = "mean")
plt.fill_between(timesteps, Gy_mean - 2* Gy_err, Gy_mean + 2* Gy_err, color='red', alpha = 0.2, label = "std")
plt.scatter(Gy_xtr, Gy_ytr, s = 5)
plt.title("Globle fitting for finger_y")
plt.xlabel("timesteps")
plt.ylabel("Predicted Value")
plt.legend()
plt.savefig("plots/globlefitting_finger_y.png")

plt.figure(9)
plt.plot()
plt.plot(timesteps, Gz_mean, '-', color = 'g',label = "mean")
plt.fill_between(timesteps, Gz_mean - 2* Gz_err, Gz_mean + 2* Gz_err, color='g', alpha = 0.2, label = "std")
plt.scatter(Gz_xtr, Gz_ytr, s = 5)
plt.title("Globle fitting for finger_z")
plt.xlabel("timesteps")
plt.ylabel("Predicted Value")
plt.legend()
plt.savefig("plots/globlefitting_finger_z.png")

# Local Posterior

for i in range(n_steps):
    if i == 0:
        time = timesteps[i * slide: i * slide + window]
        mean = Wx_mean[i, :]
        err = Wx_err[i, :]
    else:
        temp = timesteps[i * slide: i * slide + window]
        time = temp[89:99]
        mean = Wx_mean[i, 89:99]
        err = Wx_err[i, 89:99]

    plt.figure(10)
    plt.plot()
    plt.plot(time, mean, '-', color='b')
    plt.fill_between(time, mean - 2 * err, mean + 2 * err, color='b', alpha=0.2)
    plt.xlabel("time")
    plt.ylabel("prediction")
    plt.title("local fitting for finger_x")
plt.scatter(Gx_xtr, Gx_ytr, s=5)
plt.savefig("plots/local_fitting_finger_x.png")

for i in range(n_steps):
    if i == 0:
        time = timesteps[i * slide: i * slide + window]
        mean = Wy_mean[i, :]
        err = Wy_err[i, :]
    else:
        temp = timesteps[i * slide: i * slide + window]
        time = temp[89:99]
        mean = Wy_mean[i, 89:99]
        err = Wy_err[i, 89:99]

    plt.figure(11)
    plt.plot()
    plt.plot(time, mean, '-', color='r')
    plt.fill_between(time, mean - 2 * err, mean + 2 * err, color='r', alpha=0.2)
    plt.xlabel("time")
    plt.ylabel("prediction")
    plt.title("local fitting for finger_y")
plt.scatter(Gy_xtr, Gy_ytr, s=5)
plt.savefig("plots/local_fitting_finger_y.png")

for i in range(n_steps):
    if i == 0:
        time = timesteps[i * slide: i * slide + window]
        mean = Wz_mean[i, :]
        err = Wz_err[i, :]
    else:
        temp = timesteps[i * slide: i * slide + window]
        time = temp[89:99]
        mean = Wz_mean[i, 89:99]
        err = Wz_err[i, 89:99]

    plt.figure(12)
    plt.plot()
    plt.plot(time, mean, '-', color='g')
    plt.fill_between(time, mean - 2 * err, mean + 2 * err, color='g', alpha=0.2)
    plt.xlabel("time")
    plt.ylabel("prediction")
    plt.title("local fitting for finger_z")
plt.scatter(Gz_xtr, Gz_ytr, s=5)
plt.savefig("plots/local_fitting_finger_z.png")


# PLotting parameters - seperate params
def plotparam(gp, hp, n_steps, clr, tlt):

    plt.plot(timesteps, np.full(len(timesteps), np.sqrt(gp)), color=lighten_color(clr, 0.2), label='global')
    plt.plot(timesteps, np.sqrt(hp), color=lighten_color(clr, 1), label='local')
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("parameter")
    plt.title(tlt)
    plt.savefig("plots/" + tlt + ".png")


# x plots
plotparam(Gx_params[0], x_sigy, n_steps, 'b', "sigma_y_for_finger_x" )
plotparam(Gx_params[1], x_l, n_steps, 'r', "l_for_finger_x" )
plotparam(Gx_params[2], x_sign, n_steps, 'g', "sigma_n_for_finger_x" )

plotparam(Gy_params[0], y_sigy, n_steps, 'b', "sigma_y_for_finger_y" )
plotparam(Gy_params[1], y_l, n_steps, 'r', "l_for_finger_y" )
plotparam(Gy_params[2], y_sign, n_steps, 'g', "sigma_n_for_finger_y" )


plotparam(Gz_params[0], z_sigy, n_steps, 'b', "sigma_y_for_finger_z" )
plotparam(Gz_params[1], z_l, n_steps, 'r', "l_for_finger_z" )
plotparam(Gz_params[2], z_sign, n_steps, 'g', "sigma_n_for_finger_z" )


print("Process completed!")