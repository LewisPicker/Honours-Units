#!/usr/bin/env python
# coding: utf-8

# In[373]:


#Nessecary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from chainconsumer import ChainConsumer
import corner
import os
import logging
import pickle
import stan


# # Question 1

# In[374]:


#create a grid from -2 to 2 in both directions with 250 points
x = np.linspace(-2, 2, 250)
y = np.linspace(-2, 2, 250)

xx, yy = np.meshgrid(x, y)


#define the func
def rosen(x,y):
    return (1-x)**2 +100*(y - x**2)**2

#evaluate it 
z = rosen(xx,yy)

#plot the contours 
fig, ax = plt.subplots()
contour = ax.contour(xx, yy, np.log(z), levels=10, cmap='coolwarm')
fig.colorbar(contour)
ax.set_title("Log Contours of the Rosenbrock's Function")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()


# # Question 2 

# In[375]:


#prepare to record theta. 
#such that theta is appended to a list when called
#use this in the optimisation. 


thetas = []
def objective_function(theta):
    thetas.append(theta)
    return rosen(*theta)


# In[376]:


#define method and colours used in the loop
methods = ['L-BFGS-B', 'Nelder-Mead', 'SLSQP']
colour = ['tab:red', 'tab:blue', 'tab:green']

#set up the background of the plot 
fig, ax = plt.subplots()
contour = ax.contour(xx, yy, np.log(z), levels=10, cmap='coolwarm')
fig.colorbar(contour)
ax.set_title("Log Contours of the Rosenbrock's Function")
ax.set_xlabel("x")
ax.set_ylabel("y")

#loop optimisation for each method
for i in np.arange(3): 
    #optimise for initial position (-1,-1)
    thetas = []
    result = op.minimize(
        objective_function,
        [-1,-1],
        method=methods[i],
        bounds=[
            (None, None),
            (None, None)
        ]
    )
    minimum = rosen(*result.x)
    #print the results
    print(f'The minimum of the Rosenbrocks function using the {methods[i]} was found to be {minimum} corresponding to the points {result.x} (x,y)')
    thetas = np.array(thetas).T
    #plot the paths
    ax.plot(thetas[0], thetas[1], label=methods[i], c=colour[i])
    
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.legend()    
plt.show()    


# # Question 3 

# The code below generates fake data that is drawn from 
# $$
# y \sim \mathcal{N}\left(\theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3,\sigma_{y}\right)
# $$

# In[377]:


# this generates random data 

np.random.seed(0)
N = 30
x = np.random.uniform(0, 100, N)
theta = np.random.uniform(-1e-3, 1e-3, size=(4, 1))
# Define the design matrix.
A = np.vstack([
np.ones(N),
x,
x**2,
x**3
]).T
y_true = (A @ theta).flatten()
y_err_intrinsic = 10 # MAGIC number!
y_err = y_err_intrinsic * np.random.randn(N)
y = y_true + np.random.randn(N) * y_err
y_err = np.abs(y_err)


# Now assume that the data was generated from each of the following model:
# $$
# y \sim \mathcal{N}\left(\theta_0 ,\sigma_{y}\right)
# $$
# 
# $$
# y \sim \mathcal{N}\left(\theta_0 + \theta_1x ,\sigma_{y}\right)
# $$
# 
# $$
# y \sim \mathcal{N}\left(\theta_0 + \theta_1x +\theta_2x^2,\sigma_{y}\right)
# $$
# 
# $$
# y \sim \mathcal{N}\left(\theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3,\sigma_{y}\right)
# $$
# 
# $$
# y \sim \mathcal{N}\left(\theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \theta_4x^4,\sigma_{y}\right)
# $$
# 
# 
# $$
# y \sim \mathcal{N}\left(\theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \theta_4x^4 +\theta_5x^5,\sigma_{y}\right)
# $$

# Recal that the linear algebra solution follows:
# $$
# \newcommand{\transpose}{^{\scriptscriptstyle \top}}
# \mathbf{X} = \left[\mathbf{A}\transpose\mathbf{C}^{-1}\mathbf{A}\right]^{-1}\left[\mathbf{A}\transpose\mathbf{C}^{-1}\mathbf{Y}\right] \quad .
# $$
# 
# Where 
# 
# $$
# \mathbf{Y} = \left[\begin{array}{c}
#             y_{1} \\
#             y_{2} \\
#             \cdots \\
#             y_N \end{array}\right]\\
# \mathbf{A} = \left[\begin{array}{cc}
#         1 & x_1 & x_1^2 & \cdots & x_1^n \\
#         1 & x_2 & x_2^2 & \cdots & x_2^n \\
#         \cdots  & \cdots & \cdots & \ddots & \cdots \\
#         1 & x_N & x_N^2 & \cdots & x_N^n
#         \end{array}\right]\\        
# \mathbf{C} = \left[\begin{array}{cccc}
#         \sigma_{y1}^2 & 0 & \cdots & 0 \\
#         0 & \sigma_{y2}^2 & \cdots & 0 \\
#         0 & 0 & \ddots & 0 \\
#         0 & 0 & \cdots & \sigma_{yN}^2 
#         \end{array}\right] \\        
# $$
# 
# The design matrix $\mathbf{A}$ will depend on the chosen model
# 
# First lets just plot the data

# In[378]:


plt.errorbar(x, y, yerr=y_err, fmt='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data generated from a cubic model')


# In[379]:


Y = np.atleast_2d(y).T
C = np.diag(y_err * y_err)
C_inv = np.linalg.inv(C)

    
#define the design matrix for each model. 
A1 = np.vstack([np.ones_like(x)]).T
A2 = np.vstack([np.ones_like(x), x]).T
A3 = np.vstack([np.ones_like(x), x, x**2]).T
A4 = np.vstack([np.ones_like(x), x, x**2, x**3]).T
A5 = np.vstack([np.ones_like(x), x, x**2, x**3, x**4]).T
A6 = np.vstack([np.ones_like(x), x, x**2, x**3, x**4,x**5]).T

#combine
Atot = [A1,A2,A3,A4,A5,A6]

linalg_theta = []
for A in Atot:
    
    G = np.linalg.inv(A.T @ C_inv @ A)
    X = G @ (A.T @ C_inv @ Y)
    linalg_theta.append(X.T)
    print(X.T)

#we need to defin the log-liklihood for each model     


# The Bayesian Information Critera is defined by 
# 
# $$
# BIC = Dlog(N) - 2log \hat{\mathcal{L}}(\mathbf{y} | \hat{\theta})
# $$
# 
# 
# Where D is the number of model parameters, N is the is the number of data points. $\hat{\mathcal{L}}(\mathbf{y} | \hat{\mathbf{\theta}})$ is the maximum log-likelihood of the model. 
# 
# 

# In[380]:


#define the log-liklihoods of each model

def ln_likelihood(theta, x, y, y_err, i):
    if i == 0:
        b = theta
        return -0.5 * np.sum((y - b)**2 / y_err**2)
    elif i==1:
        b, a1 = theta
        return -0.5 * np.sum((y - a1 * x - b)**2 / y_err**2)
    elif i ==2:
        b, a1, a2 = theta
        return -0.5 * np.sum((y - a2*x**2 - a1 * x - b)**2 / y_err**2)
    elif i ==3:
        b, a1, a2, a3 = theta
        return -0.5 * np.sum((y - a3*x**3 - a2*x**2 - a1 * x - b)**2 / y_err**2)
    elif i == 4:
        b, a1, a2, a3, a4 = theta
        return -0.5 * np.sum((y - a4*x**4 - a3*x**3 - a2*x**2 - a1 * x - b)**2 / y_err**2)
    elif i ==5: 
        b, a1, a2, a3, a4, a5 = theta
        return -0.5 * np.sum((y - a5*x**5 - a4*x**4 - a3*x**3 - a2*x**2 - a1 * x - b)**2 / y_err**2)




# In[381]:


# now create a list of BIC with each model

BIC = []

#list of the number of model parameters 
D = np.arange(1,7,1)
#number of data points 
N = len(x)

for i in np.arange(6):
    max_log_l = ln_likelihood(linalg_theta[i].T,x,y,y_err,i)
    BIC.append(D[i]*np.log(N) - 2*max_log_l)
    
print(BIC)

plt.plot(D,np.log(BIC)) 
plt.xlabel('Model Parameters')
plt.ylabel('$log(BIC)$')
plt.show()
 


# It is easy to increase the maximum log-likelihood of a model by increasing the number of paramters. The BIC is designed to take into account this issue when comparing models and lower BIC implies a better model!
# 
# In our case the model with 4 parameters is favoured which is consitent with the generated data
# However typically we would need lower BIC to be confident that such a model is indeed favoured.

# # Question 4 

#  The data is drawn from a 2D gaussian centered around a single point with uncorrelated x and y uncertainties. 
# Such that the log likelihood of the model is: 
# 
# $$
# \log{\mathcal{L}} \propto -\sum_{i=1}^{N}\frac{\left[y_i - \hat{y}_i\right]^2}{2\sigma_{yi}^2} + \frac{\left[x_i - \hat{x}_i\right]^2}{2\sigma_{xi}^2} + log(\sigma_{yi}\sigma_{xi})
# $$
# Such that 
# $$
# U = -\log{\mathcal{L}}\propto \sum_{i=1}^{N}\frac{\left[y_i - \hat{y}_i\right]^2}{2\sigma_{yi}^2} + \frac{\left[x_i - \hat{x}_i\right]^2}{2\sigma_{xi}^2} + log(\sigma_{yi}\sigma_{xi})
# $$
# and 
# $$
# \frac{dU}{d\hat{x}_i} \propto -\sum_{i=1}^{N} \frac{x_i - \hat{x}_i}{\sigma_{xi}^2}
# $$
# 
# $$
# \frac{dU}{d\hat{y}_i} \propto -\sum_{i=1}^{N} \frac{y_i - \hat{y}_i}{\sigma_{yi}^2}
# $$

# In[382]:


#define the log-likelihood of the model
def ln_likelihood(theta, x, y, x_err, y_err):
    
    mu_x, mu_y = theta
    
    return -np.sum(np.log(y_err*x_err) + (x-mu_x)**2/(2*x_err**2) + (y-mu_y)**2/(2*y_err**2))

#define the log-prior
#perhaps we dont need to define the prior
def ln_prior(theta):
    mu_x, mu_y = theta
    if not (1 > mu_x > 0)\
    or not (1 > mu_y > 0):
        return -np.inf
    return 1.0
    
def ln_probability(theta, x, y, x_err, y_err):
    return  ln_likelihood(theta, x, y, x_err, y_err) 

def U(theta, x, y, x_err,y_err):
    return - ln_probability(theta, x, y, x_err,y_err)

def dU_dx(theta, x, y,  x_err, y_err):
    
    mu_x, mu_y = theta
    dU_dmux = -np.sum((x - mu_x)/x_err**2) 
    dU_dmuy = -np.sum((y - mu_y)/y_err**2)
    
    return np.array([dU_dmux, dU_dmuy])
    


# # Question 5

# In[384]:


def leapfrog_integration(theta, p, dU_dx, n_steps, step_size):
    """
    Integrate a particle along an orbit using the Leapfrog integration scheme.
    """
    
    #append initial positions and enrgy
    total_energy = U(theta, x, y, x_err,y_err) + p.T@p/2
    energy.append(total_energy)#record positions and total energy
    positions.append(theta)
    
    theta = np.copy(theta)
    p = np.copy(p)
    # Take a half-step first.
    p -= 0.5 * step_size * dU_dx(theta, x, y,  x_err, y_err)
    for step in range(n_steps):
        theta += step_size * p
        p -= step_size * dU_dx(theta, x, y,  x_err, y_err)
            #append intermediate positions and enrgy
        total_energy = U(theta, x, y, x_err,y_err) + p.T@p/2
        
        energy.append(np.copy(total_energy))#record positions and total energy
        positions.append(np.copy(theta))
        
        
    theta += step_size * p
    p -= 0.5 * step_size * dU_dx(theta, x, y,  x_err, y_err)
    
      #append intermediate positions and enrgy
    total_energy = U(theta, x, y, x_err,y_err) + p.T@p/2
    energy.append(total_energy)#record positions and total energy
    positions.append(theta)
    
    return (theta, -p)


# In[385]:


#define data
x, y, x_err, y_err =  np.array(
    [[0.38, 0.32, 0.26, 0.01],
    [0.30, 0.41, 0.07, 0.02],
    [0.39, 0.25, 0.09, 0.04],
    [0.30, 0.29, 0.07, 0.10],
    [0.19, 0.32, 0.23, 0.02],
    [0.21, 0.37, 0.15, 0.01],
    [0.28, 0.31, 0.01, 0.06],
    [0.24, 0.32, 0.02, 0.06],
    [0.35, 0.29, 0.15, 0.02],
    [0.23, 0.26, 0.15, 0.02]]
        ).T

plt.errorbar(x,y,xerr=x_err,yerr=y_err,fmt='o', capsize=3)


# In[386]:


initial_theta = [(2.0,2.0),(0.5,0.5),(0.1,0.1)]
N_list = [100,500,1000]

# np.random.seed(1)
p = np.random.normal(size = 2) #draw initial momentum 

print(p)


for i in range(3):
    positions = []
    energy = []

    leapfrog_integration(initial_theta[i], p, dU_dx, N_list[i], 0.001)

    dom = range(len(energy))
    plt.title(f'Initial position $mu_x$ = $mu_y$ = {initial_theta[i]}, N_steps = {N_list[i]}')
    plt.plot(dom,energy)
    plt.ylabel('Total energy')
    plt.xlabel('Integration step')
    plt.show()

    mux, muy = np.array(positions).T

    plt.plot(mux,muy)
    plt.xlabel('$\mu_x$')
    plt.ylabel('$\mu_y$')
    plt.show()


# The Leap-frog integrator is a time-reversible and volume-preserving scheme. Hence the hamiltonian remains constant and total energy is therefore conserved. \
# This is why leapfrog is used in conjunction with the Hamiltonian MC, since other schemes such as RK4 will not conserve energy and cause HMC to fail. 
# 

# # Question 6 
# 

# In[387]:


#defining the hamiltonian MCMC
def h_mcmc(theta, N):
    # step size
    dx = 0.01
    # no. of steps
    L = 10
    # initial guess
    theta0 = theta
    chain = []
    for i in range(N):
        print(f"Running step {i} of {N}", end='\r')
        # 1. draw from momentum distribution.
        p0 = np.random.normal(size = 2)
        # 2. integrate for L steps.
        theta1, p1 = leapfrog_integration(theta0, p0, dU_dx, L, dx)
        p1 = -p1

        alpha = np.exp(-U(theta1, x, y, x_err, y_err) + U(theta0, x, y, x_err, y_err) - p1.T@p1/2 + p0.T@p0/2)
        u = np.random.uniform(0., 1.)
        if alpha >= u:
            # accept
            theta0 = theta1
        chain.append(theta0)
    return np.array(chain)


# In[395]:


initial_theta = [(0.2,0.2),(2.0,2.0)]
colour = ['r', 'g']

# chain_list = []
# for i in range(2):                 
#     chain = h_mcmc(initial_theta[i], 2000)
#     chain_list.append(chain) 
    
fig, ax = plt.subplots(figsize=(10, 3))
ax.legend(loc = 'right')
ax.set_ylabel(r"$\mu_x, \mu_y$")
ax.set_xlabel("Steps ")
ax.set_ylim(0.2,0.5)
plt.title(f'Hmcmc resutls for initial guesses {initial_theta[0]} and {initial_theta[1]}')
    
for i in range(2):    
    mux, muy = chain_list[i].T 
    mean_mux = np.mean(mux)
    mean_muy = np.mean(muy)

    mean_x = np.full(len(mux),mean_mux)
    mean_y = np.full(len(muy),mean_muy)

    ax.plot(mux, c=colour[i], label="$\mu_y$ HMC chain", alpha = 0.5)
    ax.plot(muy, c=colour[i], label="$\mu_y$ HMC chain", alpha = 0.5)
    ax.plot(mean_x, c='k', label="Mean $\mu_y$",linestyle = '--')
    ax.plot(mean_y, c='k', label="Mean $\mu_y$", linestyle = '--')
    plt.axvline(x=1000, color='blue', linestyle='--', alpha = 0.3, label= 'burn')
    
legend_elements = [
    plt.Line2D([0], [0], color='k', lw=2, label='mean', linestyle = '--'),
    plt.Line2D([0], [0], color='r', lw=2, label='$\mu_x$, $\mu_y$'),
    plt.Line2D([0], [0], color='g', lw=2, label='$\mu_x$, $\mu_y$'),
    plt.Line2D([0], [0], color='blue', lw=2, label='burn',linestyle = '--', alpha = 0.3)
]

ax.legend(handles=legend_elements)
fig.tight_layout()

c = ChainConsumer()
for i in range(2):
    c.add_chain(chain_list[i][1000:], parameters=["$\mu_x$", "$mu_y$"])
fig = c.plotter.plot(filename="example.png", figsize="column")

#     # plot corner plot
#     fig = corner.corner(chain, labels=["$\mu_x$", "$\mu_y$"], show_titles=True, title_kwargs={"fontsize": 12})


#     for ax in fig.get_axes():
#         ax.tick_params(axis='both', size=3)
#         ax.set_title(ax.get_title())
#         ax.title.set_position([0.5, 0.95])
#     plt.show()


# The convergence of the chains that had different initialisation points is a promising sign that the HMC shceme is working corectly. Although we see a general 'bumbyness' in the guassian curves this could be smoothened out with increasing more steps. The predicted values for the model parameters are also consistent to our expectations bye 'eye'. We could furthur improve this scheme if we were to tune the parameters for each step allowing for a more effective sampler.  

# # Quesiton 7

# In[ ]:


# ###############THIS IS MY .stan FILE ################

# data {
#     int<lower=1> N_data; // number of data points

#     // x-values of the data is uncertain.
#     vector[N_data] x;
#     vector[N_data] sigma_x;

#     // y-values of the data is uncertain.
#     vector[N_data] y;
#     vector[N_data] sigma_y;

# }

# parameters {
#     // Mean value of the x-guassian.
#     real<lower=-2.0, upper=2.0> mu_x;

#     // Mean value of the y-guassian.
#     real<lower=-2.0, upper=2.0> mu_y;

# }

# model {
#     for (i in 1:N_data) {
#         x[i] ~ normal(
#           mu_x,  
#           sigma_x[i]
#         );
# 	y[i] ~ normal(
#            mu_y, 
#           sigma_y[i]
#         );
#     }
# }


# In[329]:


from cmdstanpy import CmdStanModel, install_cmdstan
install_cmdstan()


# In[396]:


model = CmdStanModel(stan_file='PS2-1.stan')

# Data.
data = dict(
    N_data = 10,
    x=x,
    y=y,
    sigma_x = x_err,
    sigma_y = y_err,
)

fit1 = model.sample(data = data)
fit2 = model.sample(data = data)


#run samples


# In[397]:


plt.xlabel('HMC step')
plt.ylabel('$\mu_x$, $\mu_y$')
plt.plot(fit1.mu_x, c = 'r', label = '$\mu_x$ fit 1')
plt.plot(fit1.mu_y, c = 'r',label = '$\mu_y$ fit 1')
plt.plot(fit2.mu_x, c = 'c',label = '$\mu_x$ fit 2')
plt.plot(fit2.mu_y, c = 'y',label = '$\mu_y$ fit 2')
plt.legend()
plt.show()


# In[398]:


chain1 = [fit1.mu_x,fit1.mu_y]
chain2 = [fit2.mu_x, fit2.mu_y]
c = ChainConsumer()
c.add_chain(chain1, parameters=["$\mu_x$","$\mu_y$"])
c.add_chain(chain2, parameters=["$\mu_x$","$\mu_y$"])
fig = c.plotter.plot(filename="example.png", figsize="column")


# As one might of expected the results look alot smoother than my scheme. This is because Stan tunes paramters in each step such as the number of steps and step size for the intergration. This allows for a more effective and faster sampler. 

# In[ ]:




