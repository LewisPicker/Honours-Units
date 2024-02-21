from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

aj =5.2
asat = 9.54
gamma = (0.0003/0.001)*(asat/aj)**(1/2)
ms = 0.0003
mj = 0.001
pj = 11.87

def f(t,r): #these are the equations of motion
    eb,wb,ec,wc = r
    feb = -(5/4)*(aj/asat)*(ec)*np.sin(wb-wc)
    fwb = 1 - (5/4)*(aj/asat)*(ec/eb)*np.cos(wb-wc)
    fec = (5/4)*(1/gamma)*(aj/asat)*eb*np.sin(wb-wc)
    fwc = (1/gamma)*(1-(5/4)*(aj/asat)*(eb/ec)*np.cos(wb-wc))
    return feb, fwb, fec, fwc


sol = integrate.solve_ivp(f,(0,4.5),(0.048,14.73*np.pi/180,0.054,92.60*np.pi/180),method='RK45', t_eval = np.linspace(0,4.5,100))

eb, wb, ec, wc = sol.y
t = sol.t*(4/3)*(pj/(2*np.pi))*(1/ms)*(asat/aj)**3  #converting into years
dw = np.mod((wb-wc)*(180/np.pi),360)





plt.plot(t, eb)
plt.plot(t, ec)
plt.xlabel('Time (years)')
plt.ylabel('Ecentricity')
plt.title('Secular evolution of the ecentricity')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.show()

plt.plot(t, dw)
plt.xlabel('Time (years)')
plt.ylabel('wj - ws (deg)')
plt.title('Secular evolution of the ecentricity')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.show()
