import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sympy
import pdb

#constants
rsol = 6.957e10 #CGS
msol = 1.989e33 #CGS
lsol = 3.8e33   #cgs

G = 6.67e-8 #CGS
e_charge = 4.8e-10 #CGS electron charge
kb = 1.38e-16 #CGS boltzman constant
Na = 6.02e23 #advagadros number
amu = 1.66e-24 #g atomic mass unit
h = 6.63e-27  #planks constant cgs
me = 9.1e-28 #electron mass grams
c = 3e10 # speed of light CGS
rad_constant = 7.56e-15 #CGS radiation constant



# dU/dZeta function
def f_u(zeta, u, theta):
    if zeta == 0:
        return -1/3 #initial condition
    return -2*u/zeta - theta**n
# dTheta/dZeta == u
def f_theta(zeta, u, theta):
    return u
# simple test function
def y_func(x,y,th):
    return y

# this function defines RK4 and has options to solve a coupled ODE
# you can chouse the function (f)
def RK4(f, zeta0, u0, theta0, var):
    zeta = zeta0 + h
    if var == 'u':
        k1 = f(zeta0, u0, theta0)
        k2 = f((zeta0+h/2), (u0+h*k1/2), theta0)
        k3 = f((zeta0+h/2), (u0+h*k2/2), theta0)
        k4 = f((zeta0+h), (u0+h*k3), theta0)
        k = (k1+2*k2+2*k3+k4)*h/6
        u = u0 + k
        return zeta, u
    elif var == 'theta':
        k1 = f(zeta0, u0, theta0)
        k2 = f((zeta0+h/2), (u0), theta0 +h*k1/2)
        k3 = f((zeta0+h/2), (u0), theta0 +h*k2/2)
        k4 = f((zeta0+h), (u0), theta0+h*k3)
        k = (k1+2*k2+2*k3+k4)*h/6
        theta = theta0 + k
        return zeta, theta
    else:
        print("WARNING")


#step size
h = 0.01
#polytrope index
n = 1.5
#number of steps
n_steps = 2000

#n_list = [0,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
#looping over all polytropes can make a list.


# for n in [0,1.5,2,2.5,3,3.5,4,4.5,5]:
#initial conditions
zeta = 0.0
theta = 1.0
u = 0.0
theta_sol = []
zeta_sol = []
u_sol = []
for i in range(n_steps):
    zeta_tmp, u = RK4(f_u,zeta, u, theta, 'u')
    zeta, theta = RK4(f_theta,zeta, u, theta, 'theta')
    zeta = zeta_tmp
    u_sol.append(u)
    theta_sol.append(theta)
    zeta_sol.append(zeta)
    if theta < 0: #zeta_1 values
        break # we are only interesting in realistic solutions
print('zeta1',zeta_sol[-1], 'u1', u_sol[-1])

# # plt.plot(zeta_sol,u_sol, label = 'u')
# plt.plot(zeta_sol,theta_sol, label = 'n = '+ str(n))
# plt.plot(zeta_sol, [0 for n in range(len(zeta_sol))], linestyle = '--', color = 'b')
# plt.xlabel('zeta')
# plt.ylabel('theta')
# plt.legend()
# plt.show()


##################### Solutions for Q1 a) ##############
# zeta1 = zeta_sol[-1]
# u1 = u_sol[-1]
# rho_c = 160 #g/cm3
# M = msol
# mu = 0.61
#
# alpha = (M/(-4*np.pi*rho_c*u1*zeta1**2))**(1/3)
# K = ((alpha**2)*4*np.pi*G*rho_c**((n-1)/n))/(n+1)
# print('alpha = ', "{:.2e}".format(alpha),'K = ', "{:.2e}".format(K))
#
# R = alpha*zeta1
# P_c = K*rho_c**((n+1)/n)
# T_c = (P_c*mu*amu)/(rho_c*kb)
#
# print('R = ', "{:.2e}".format(R), 'P_c = ', "{:.2e}".format(P_c), 'T_c = ', "{:.2e}".format(T_c))

# zeta1 3.649999999999966 u1 -0.2037179903573512
# alpha =  7143262471.188932 K =  92874302746636.34
# R =  0.3747722871904465 (rsol), R =  2.61e+10 P_c =  4.38e+17 T_c =  2.01e+07

# actual value of the sun T_c = 15e6K
# pressure = 265e9 bar (Si) = 265e15 = 2.65e17 barye (cgs)

#r = a *zeta
#a**2 = (n+1)K/(4*pi*G*rho_c**((n-1)/n))
#M = -4*pi*alpha**3*rho_c*z_1**2 *u1
# P =kb*T*rho/(mu*mu)

################## Solution for Q1 b) ################
# zeta1 = zeta_sol[-1]
# u1 = u_sol[-1]
# rho_c = 6 #g/cm3
# M = 25*msol
# mu = 0.61
#
# alpha = (M/(-4*np.pi*rho_c*u1*zeta1**2))**(1/3)
# K = ((alpha**2)*4*np.pi*G*rho_c**((n-1)/n))/(n+1)
# print('alpha = ', "{:.2e}".format(alpha),'K = ', "{:.2e}".format(K))
#
# R = alpha*zeta1
# P_c = K*rho_c**((n+1)/n)
# print(kb*rho_c/(mu*amu), rad_constant/3, P_c )
#
# print(kb*5.62e7*rho_c/(mu*amu))
#
#
# def EOS(T):
#     return kb*rho_c*T/(mu*amu) + T**4*(rad_constant/3) - P_c
#
# # print(EOS(rho_c, P_c, T))
# T_c = fsolve( EOS,10)[0]
# # print(T_c[0])
# P_c_gas = kb*T_c*rho_c/(mu*amu)
# print('P_c_gas = ', "{:.2e}".format(P_c_gas))
#
# print('R = ', "{:.2e}".format(R), 'P_c = ', "{:.2e}".format(P_c), 'T_c = ', "{:.2e}".format(T_c))
# print(P_c_gas/P_c)
# exit()
############# Solution for Q2 a) ###########
#Leave everythin in terms of rho_c I guess.

#get the temp funciton
# rho_c = sympy.Symbol('rho_c')#g/cm3
# T = sympy.Symbol('T')
# kb = sympy.Symbol('kb')
# rho = sympy.Symbol('rho')
# rad_constant = sympy.Symbol('rad_constant')
# mu = sympy.Symbol('mu')
# amu = sympy.Symbol('amu')
# P = sympy.Symbol('P')
# T_sol = sympy.solve(kb*rho*T/(mu*amu) + (rad_constant/3)*T**4 - P, T)
# print(T_sol)
#it was the second peicewise
# exit()

X1 = 0.715 #mass fraction of H
zeta_sol = np.array(zeta_sol)
theta_sol = np.array(theta_sol)
zeta1 = zeta_sol[-1]
u1 = u_sol[-1]
rho_c = sympy.Symbol('rho_c')
M = msol
mu = 0.61

alpha = (M/(-4*np.pi*rho_c*u1*zeta1**2))**(1/3)
K = ((alpha**2)*4*np.pi*G*rho_c**((n-1)/n))/(n+1)
rho = rho_c*theta_sol**n
P = K*rho**(5/3)
print('n =' ,n)
# print(rho[-10],P[-10])

print('K = ', K, 'alpha = ', alpha)

#
def temperature(P, rho):
    T = (-sqrt(-2*P/(a*(9*R**2*rho**2/(16*a**2*mu**2) + sqrt(P**3/a**3 + 81*R**4*rho**4/(256*a**4*mu**4)))**(1/3)) + 2*(9*R**2*rho**2/(16*a**2*mu**2) \
    + sqrt(P**3/a**3 + 81*R**4*rho**4/(256*a**4*mu**4)))**(1/3))/2 + sqrt(2*P/(a*(9*R**2*rho**2/(16*a**2*mu**2) + sqrt(P**3/a**3 + 81*R**4*rho**4/(256*a**4*mu**4)))**(1/3)) \
    + 6*R*rho/(a*mu*sqrt(-2*P/(a*(9*R**2*rho**2/(16*a**2*mu**2) + sqrt(P**3/a**3 + 81*R**4*rho**4/(256*a**4*mu**4)))**(1/3)) + 2*(9*R**2*rho**2/(16*a**2*mu**2)\
    + sqrt(P**3/a**3 + 81*R**4*rho**4/(256*a**4*mu**4)))**(1/3))) - 2*(9*R**2*rho**2/(16*a**2*mu**2) + sqrt(P**3/a**3 + 81*R**4*rho**4/(256*a**4*mu**4)))**(1/3))/2
    return T

for rho_c in range(130,200,1):

T_list = []
for i in range(len(rho)):
    print(i)
    T = Temperature(P[i],rho[i])
    T_list.append(T)

T_sol_list = np.array(T_sol_list)

T9_sol = T_sol_list*(10**(-9))

g11 = (1+3.82*T9_sol +1.51*T9_sol**2 +0.144*T9_sol**3 - 0.0144*T9_sol**4)
epp = 5.14e4 * g11 * rho[356:] * X1**2 * T9_sol**(-2/3) * np.exp(-3.381/(T9_sol**(1/3)))












###############################################

#solveing the exponential function
# for i in range(100):
#     x,y = RK4(y_func,x,y,5,var = 'u')
#     y_sol.append(y)
#     x_sol.append(x)
#
# plt.plot(x_sol,y_sol)
# plt.show()
