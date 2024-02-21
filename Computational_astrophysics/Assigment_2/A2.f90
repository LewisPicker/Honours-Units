program main
    use init, only:initialise
    use step, only: leap_frog, r_k_4
    use out, only: output

    implicit none
    integer, parameter:: n = 5340
    real:: x,y,v_x,v_y,a_x,a_y
    real :: e, dt, t
    integer:: n_print,i, scheme

    !define scheme
    print*, 'choose scheme leapfrog(0), RK4(1)'
    read*, scheme
    !set eccentricity
    print*, 'choose eccentricity (0-1)'
    read*, e
    !set timestep
    print*,"choose timestep (preferably 0.01 - 0.1)"
    read*, dt
    !print after nprint steps
    n_print = 10
    !set initial time
    t=0
    call initialise(x, y, v_x, v_y, e)

    call output(x,y,t)
    open(1, file = 'angularP_Energy', status = 'replace', action ='write')
    write(1,*) '# t, angular momentum, enery'
    open(3, file = 'trajectory', status = 'replace', action ='write')
    write(3,*) '#  x,  y,  t'

    do i = 1, n
        if (scheme == 0) then
            call leap_frog(x,y,v_x,v_y,a_x,a_y, t, dt)
            if (i>n_print) then
                n_print = n_print + 10
                !if you want to show the motion
                call output(x,y,t)
                print*, "t = ",t
            endif
        elseif (scheme == 1) then
            call r_k_4(v_x,v_y,x,y,t,dt)
            if (i>n_print) then
                n_print = n_print + 10
                !if you want to show the motion
                call output(x,y,t)
                print*, "t = ",t
            endif
        endif


        !write the trajectory
        write(3,*) x,y,t
        !write the angular momentum and total energy
        write(1,*) t,(x*v_y - y*v_x),0.5*(v_x**2+v_y**2) - 1/(x**2+y**2)*0.5
    enddo


    print*,'done'
end program
