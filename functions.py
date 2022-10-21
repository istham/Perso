## NOTE: we have to impport everything needed in this file as well, otherwsie pd, np, plt
## are not defined, even if they are in the notebook
import os
import random
import pandas as pd
import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def _fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def unicycle(t, x):
    
    # paramater values
    v = 0.5
    omega = -math.pi / 3 

    dp0_dt = v * np.cos(x[2])
    dp1_dt = v * np.sin(x[2])
    dphi_dt = omega
    
    return np.array([dp0_dt, dp1_dt, dphi_dt])

def generate_trajectories(number_trajectories, noise):
    dfs = []
    for i in range(number_trajectories):
        solution = solve_ivp(unicycle, y0=[np.random.random_sample() * 2 - 1, np.random.random_sample() * 2 - 1, np.random.random_sample() * 2 * math.pi - math.pi], t_span=[0,50], t_eval=np.arange(0,50,0.02), method="RK45")

        steps = solution.t.size
        t = solution.t
        p_0 = solution.y[0,:] + np.random.normal(loc=0, scale=noise * 2, size=steps)
        p_1 = solution.y[1,:] + np.random.normal(loc=0, scale=noise * 2, size=steps)
        phi = solution.y[2,:] + np.random.normal(loc=0, scale=noise * 2 * math.pi, size=steps)
        df = pd.DataFrame(data={'t': t, 'p_0': p_0, 'p_1': p_1, 'phi': phi})
        dfs.append(df)
        
    # Return the last trajectory as example
    return dfs

def plot_trajectories(df_, labels=None):
    
    if not isinstance(df_, list):
        df_ = [df_]
    
    plt.figure()
    for j, X in enumerate(df_):
        plt.scatter(X.iloc[:,1], X.iloc[:,2], s=1, marker=".")
    plt.xlabel("$p_0$", size=15)
    plt.xticks(size=15)
    plt.ylabel("$p_1$", size=15)
    plt.yticks(size=15)
    plt.axis('square')
    plt.grid()
    if labels is not None:
        plt.legend(labels, prop={'size':15}, markerscale=10)
    plt.show()

    plt.figure()
    for j, df in enumerate(df_):
        plt.scatter(df.iloc[:,0], df.iloc[:,3], s=1, marker=".")
    plt.ylabel("$\\varphi$", size=15)
    plt.xticks(size=15)
    plt.xlabel("$t$", size=15)
    plt.yticks(size=15)
    plt.grid()
    if labels is not None:
        plt.legend(labels, prop={'size':15}, markerscale=10)
    plt.show()

def generate_X_Y(dfs_train_, dfs_test_, generate_target):
    dfs = []
    for j in range(2):
        Xs = []
        Ys = []
        ts = []
        for i in range(len(dfs_train_) if j==0 else len(dfs_test_)):
            df = dfs_train_[i] if j == 0 else dfs_test_[i]
            X, Y, t = generate_target(df)
            Xs.append(X)
            Ys.append(Y)
            ts.append(t)

        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
        t = np.concatenate(ts)

        df = pd.DataFrame(data={'t': t, 'p0': X[:,0], 'p1': X[:,1],'phi': X[:,2], 'dp0dt': Y[:,0], 'dp1dt': Y[:,1],'dphidt': Y[:,2],})

        dfs.append(df)
        
    # return the train and test data
    return dfs

def separate_trajectories(df_test, number_trajectories):
    a = len(df_test) // number_trajectories
    
    t_test = [df_test[['t']].to_numpy()[i*a:(i+1)*a,:].flatten() for i in range(number_trajectories)]
    X_test = [df_test[['p0', 'p1', 'phi']].to_numpy()[i*a:(i+1)*a,:] for i in range(number_trajectories)]
    Y_test = [df_test[['dp0dt', 'dp1dt', 'dphidt']].to_numpy()[i*a:(i+1)*a,:] for i in range(number_trajectories)]
    
    return t_test, X_test, Y_test

def plot_data(t, X, Y, labels=None):
    
    fig, ax = plt.subplots(2, 2, figsize=(16,9))
    
    ax[0,0].scatter(X[:,0], X[:,1])
    ax[0,0].set_xlabel("$p_0$", size=15)
    ax[0,0].set_ylabel("$p_1$", size=15)
    ax[0,0].axis('square')
    ax[0,0].set_title('$\mathbf{x}_{1:2}$', size=18)
    ax[0,0].grid()
    
    ax[1,0].scatter(Y[:,0], Y[:,1])
    ax[1,0].set_xlabel("$\dot{p}_0$", size=15)
    ax[1,0].set_ylabel("$\dot{p}_1$", size=15)
    ax[1,0].axis('square')
    ax[1,0].set_title('$\mathbf{y}_{1:2}$', size=18)
    ax[1,0].grid()
    
    ax[0,1].scatter(t,  X[:,2])
    ax[0,1].set_xlabel("t", size=15)
    ax[0,1].set_ylabel("$\\varphi$", size=15)
    ax[0,1].set_title('$\mathbf{x}_{3}$', size=18)
    ax[0,1].grid()
    
    ax[1,1].scatter(t,  Y[:,2])
    ax[1,1].set_xlabel("t", size=15)
    ax[1,1].set_ylabel("$\dot{\\varphi}$", size=15)
    ax[1,1].set_title('$\mathbf{y}_{3}$', size=18)
    ax[1,1].grid()
    
    fig.tight_layout()

    plt.show()

def predict_and_plot(X_, Y_, t_, w_1, w_2, w_3, feature_map):
    
    if not isinstance(t_, list):
        t_ = [t_]
    if not isinstance(Y_, list):
        Y_ = [Y_]
    if not isinstance(X_, list):
        X_ = [X_]
    
    # define vector field
    def f_vector(x, w_1, w_2, w_3):
        return np.array([w_1.T @ feature_map(np.expand_dims(x, axis=0))[0,:], w_2.T @ feature_map(np.expand_dims(x, axis=0))[0,:], w_3.T @ feature_map(np.expand_dims(x, axis=0))[0,:]])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7+len(t_),7+len(t_)))

    for t,X,Y in zip(t_,X_,Y_):
        # integrate
        sol = solve_ivp(lambda t,x: f_vector(x, w_1, w_2, w_3), y0=X[0,:], t_span=[t[0],t[-1]], t_eval=t, method="RK45")
        Y_pred = feature_map(sol.y.T) @ np.stack([w_1, w_2, w_3], axis=1)
        
        # plotting
        ax1.plot(sol.y[0,:], sol.y[1,:], label="$(p_0, p_1)$ predicted")
        ax1.scatter(X[:,0], X[:,1],s=1, marker=".", c="k", label="$(p_0, p_1)$ data")
        
        ax2.plot(sol.t,sol.y[2,:], label="$\\varphi$ prediction")
        ax2.scatter(t, X[:,2], s=1, marker=".", c="k", label="$\\varphi$ data")
        
        ax3.plot(Y_pred[:, 0], Y_pred[:, 1], label="$(\dot{p}_0, \dot{p}_1)$ predicted")
        ax3.scatter(Y[:,0], Y[:,1],s=1, marker=".", c="k", label="$(\dot{p}_0, \dot{p}_1)$ data")
        
        ax4.plot(sol.t, Y_pred[:,2], label="$\dot{\\varphi}$ prediction")
        ax4.scatter(t, Y[:,2], s=1, marker=".", c="k", label="$\dot{\\varphi}$ data")

    # Cosmetics
    ax1.set_xlabel("$p_0$")
    ax1.set_ylabel("$p_1$")
    ax2.set_ylabel("$\\varphi$")
    ax2.set_xlabel("$t$")
    ax3.set_xlabel("$\dot{p}_0$")
    ax3.set_ylabel("$\dot{p}_1$")
    ax4.set_ylabel("$\dot{\\varphi}$")
    ax4.set_xlabel("$t$")
    
    ax1.axis('equal')
    ax2.axis('equal')
    ax3.axis('equal')
    ax4.axis('equal')
 
    if len(t_) == 1:
        ax1.legend(bbox_to_anchor=(0.5,-0.25), loc='upper center', markerscale=5)
        ax2.legend(bbox_to_anchor=(0.5,-0.25), loc='upper center', markerscale=5)
        ax3.legend(bbox_to_anchor=(0.5,-0.25), loc='upper center', markerscale=5)
        ax4.legend(bbox_to_anchor=(0.5,-0.25), loc='upper center', markerscale=5)
    
    fig.tight_layout()
    plt.show()
