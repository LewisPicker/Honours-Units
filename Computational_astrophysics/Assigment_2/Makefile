FC=gfortran
FFLAGS= -O3 -Wall -Wextra -std=f2008 -fdefault-real-8 -fcheck=all -g -fbacktrace -finit-real=NaN
SRC=  init.f90 timestep.f90 output.f90 A2.f90
OBJ=${SRC:.f90=.o}

%.o:%.f90
	$(FC) $(FFLAGS) -o $@ -c $<

main:$(OBJ)
	$(FC) $(FFLAGS) -o $@ $(OBJ)

clean:
	rm -f *.mod *.o main output_*

run:main
	./main
main.o: setup.o
density.o: kernel.o
eos.o: kernel.o
