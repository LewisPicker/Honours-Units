	parameter (n=3,m=30)
	real t(0:m),x(0:m),xd(0:m),xe(0:m)
	common/var/u(n)

	pi=4.0*atan(1.0)

	t(0)=0
	x(0)=1
	xd(0)=0

	u(1)=t(0)
	u(2)=x(0)
	u(3)=xd(0)

	t(m)=4*pi

	dt=(t(m)-t(0))/real(m)
	print*,'dt',dt

	do i=1,m
	   call rk(dt)

	   t(i)=u(1)
	   x(i)=u(2)
	   xd(i)=u(3)

	   xe(i)=t(i)+1-sin(t(i))     !  exact solution

	   error=x(i)-xe(i)

	   print*,t(i),x(i),xe(i),error
	enddo

	end

	
	subroutine deriv
	
	parameter (n=3)
	common/var/u(n)
	common/der/du(n)

	t=u(1)
	x=u(2)

	td=1
	xd=u(3)
	xdd=sin(t)

	du(1)=td
	du(2)=xd
	du(3)=xdd

	end



      subroutine rk(h)

      parameter (n=3)

      common/var/u(n)
      common/der/du(n)
      real u0(n),ut(n)

      real a(4),b(4)

      a(1)=h/2.
      a(2)=a(1)
      a(3)=h
      a(4)=0

      b(1)=h/6.
      b(2)=h/3.
      b(3)=b(2)
      b(4)=b(1)

      do i=1,n
         u0(i)=u(i)
         ut(i)=0
      enddo

      do j=1,4
          call deriv

              do i=1,n
                  u(i)=u0(i)+a(j)*du(i)
                  ut(i)=ut(i)+b(j)*du(i)
              enddo
      enddo

      do i=1,n
         u(i)=u0(i)+ut(i)
      enddo

      return
      end


