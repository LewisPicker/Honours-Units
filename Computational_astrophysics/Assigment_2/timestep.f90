module step
    implicit none
contains

    subroutine leap_frog(x,y,v_x,v_y,a_x,a_y, t, dt)
        real, intent(inout) ::x, y, v_x, v_y, a_x, a_y,dt, t
        real :: G, M, r, a_x_1, a_y_1
        G = 1.
        M = 1.
        r = sqrt(x**2+y**2)
        a_x = - x*G*M/(r**3)
        a_y = -y*G*M/(r**3)

        !calculate positions
        x = x+ dt*v_x + 0.5*a_x*dt**2
        y = y+ dt*v_y + 0.5*a_y*dt**2

        !update acceleration
        r = sqrt(x**2+y**2)
        a_x_1 = - x*G*M/(r**3)
        a_y_1 = -y*G*M/(r**3)

        !calculate velocity
        v_x = v_x + (a_x+a_x_1)*dt*0.5
        v_y = v_y + (a_y+a_y_1)*dt*0.5
        t = t + dt
    end subroutine leap_frog

    function f(u)
        real, intent(in):: u(2,2)
        real :: f(2,2)

        f(1,:) = u(2,:)
        f(2,:) = -u(1,:)/dot_product(u(1,:),u(1,:))**1.5
    end function f

    subroutine r_k_4(v_x,v_y,x,y,t,dt)
        real:: k1(2,2), k2(2,2), k3(2,2), k4(2,2), u(2,2)
        real, intent(inout):: v_x,v_y,x,y,t,dt

        u(1,:) = (/x,y/)
        u(2,:) = (/v_x,v_y/)

        k1 = f(u)
        k2 = f(u+0.5*dt*k1)
        k3 = f(u+0.5*dt*k2)
        k4 = f(u+dt*k3)

        u = u + 1./6.*dt*(k1 + 2.*k2 + 2.*k3 + k4)
        t = t + dt
        !update the positions
        x = u(1,1)
        y = u(1,2)
        v_x = u(2,1)
        v_y = u(2,2)
    end subroutine r_k_4

end module
