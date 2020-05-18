#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define SEED     921
#define NUM_ITER 1000000000


double sendCount(int rank, int size, double pi, int depth)
{
	double rec_pi;
	if(pow(2,depth)>=size){
		 return pi;	
	}
	
	if(rank%(int) pow(2,depth+1)==0){
	
		MPI_Recv(&rec_pi, 1, MPI_DOUBLE, rank+(int) pow(2,depth), 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		pi+=rec_pi;
		return sendCount(rank, size, pi, depth+1);
		
	}else{
	
		MPI_Send(&pi, 1, MPI_DOUBLE, rank-(int) pow(2,depth), 0, MPI_COMM_WORLD);
	
		return 0.0;
	}
	
}

int main(int argc, char* argv[])
{
    int count = 0;
    double x, y, z, pi, final_pi, rec_pi;
	double t1, t2;
	int rank, size, provided;
	final_pi = 0.0;
	
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	    
	t1 = MPI_Wtime();	
    srand(SEED*rank); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    // Calculate PI following a Monte Carlo method
    for (int iter = 0; iter < NUM_ITER/size; iter++)
    {
        // Generate random (X,Y) points
        x = (double)random() / (double)RAND_MAX;
        y = (double)random() / (double)RAND_MAX;
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0)
        {
            count++;
        }
    }
    
    // Estimate Pi and display the result
    pi = ((double)count / (double)NUM_ITER) * 4.0;
	
	int depth = 0;
    pi = sendCount(rank, size, pi, depth);
	
	if(rank == 0){	
		printf("The result is %f\n", pi);
		t2 = MPI_Wtime();
		printf("%f \n", t2-t1);
	}
	
	MPI_Finalize();
    return 0;
}

