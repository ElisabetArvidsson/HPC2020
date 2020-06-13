This is the README file for the MetHPC final project. 

To compile the main program:

CC -fopenmp -O3 main.cpp 

The program takes two inputs, the size of the matrix and mode:

Size of the matrix:
"-n <number>" 

Mode: 
"-c" to check the correctness or "-b" to benchmark.

example call, uses one node, size 64 and mode "benchmark":

srun -n 1 ./a.out -n 64 -b


To use the BLAS library one line in the main file needs to be uncommented. See line 358 in the main file. 

Firstly change the compiler:

module swap PrgEnv-cray PrgEnv-intel 
module swap intel intel/19.0.1.144

Compile with the following: 

CC -DMKL_ILP64 -I/cfs/klemming/pdc.vol.beskow/intel/19.0.1.144/compilers_and_libraries_2019.1.144/linux/mkl/include -L/cfs/klemming/pdc.vol.beskow/intel/19.0.1.144/compilers_and_libraries_2019.1.144/linux/mkl/lib/intel64_lin -L/cfs/klemming/pdc.vol.beskow/intel/19.0.1.144/compilers_and_libraries_2019.1.144/linux/mkl/../compiler/lib/intel64_lin -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -O3 main.cpp -o matmul_d.o 

The executable file is run with the same arguments as before.
