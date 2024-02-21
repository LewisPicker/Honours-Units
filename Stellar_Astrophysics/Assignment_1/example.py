def RK4(f, zeta0, u0, theta0, var):
    #set up step size based on number of iterations
    k1 = f(zeta0, u0, theta0)
    k2 = f((zeta0+h/2), (u0+h*k1/2), theta0 +h*k1/2)
    k3 = f((zeta0+h/2), (u0+h*k2/2), theta0 +h*k2/2)
    k4 = f((zeta0+h), (u0+k3), theta0+k3)
    k = (k1+2*k2+2*k3+k4)*h/6 #overall step for u(zeta)

    zeta = zeta0 + h
