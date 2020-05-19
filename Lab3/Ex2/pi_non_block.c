#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define SEED     921
#define NUM_ITER 1000000000

int main(int argc, char* argv[])
{
    int count = 0;
    double x, y, z, pi, final_pi, rec_pi;
	double t1, t2;
	int rank, size, provided;
	final_pi = 0.0;
	
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
	
	t1 = MPI_Wtime();	
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
    
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
	
	if (rank != 0){
		MPI_Request request;
		MPI_Isend(&pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, MPI_STATUS_IGNORE);
	}else{

		MPI_Request requests[size-1];	
		double counts[size-1];

		for(int i = 1; i <size; i++){
			MPI_Irecv(&counts[i-1], 1, MPI_DOUBLE, i, 0,MPI_COMM_WORLD, &requests[i-1]);
		}

		MPI_Waitall(size-1, requests, MPI_STATUSES_IGNORE);
		final_pi=pi;
		for(int i = 0; i<size-1; i++)
		{
			final_pi+=counts[i];
		}
	
    	printf("The result is %f\n", final_pi);
		t2 = MPI_Wtime();
		printf("%f \n", t2-t1);
	}
	
	MPI_Finalize();
    return 0;
}

