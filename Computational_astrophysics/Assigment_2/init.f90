module init
    implicit none
contains

    subroutine initialise(x, y, v_x, v_y, e)
        real, intent(inout)::x, y, e, v_x, v_y
        x = 1. -e
        y = 0.
        v_x = 0.
        v_y = sqrt((1.+e)/(1.-e))
    end subroutine initialise

end module
