
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define SEED     921
#define NUM_ITER 10000000000

int main(int argc, char* argv[])
{
	unsigned long count = 0;
	double x, y, z;
	unsigned int seed;

	double t = omp_get_wtime();

	#pragma omp parallel private(x,y,z,seed) reduction(+:count)
	{

		// int partial = 0;
		int threads = omp_get_num_threads();
		int rank = omp_get_thread_num();

		seed = SEED^rank;
		// srand(SEED+rank); // Important: Multiply SEED by "rank" when you introduce MPI!
		
		// Calculate PI following a Monte Carlo method
		for (unsigned long iter = 0; iter < NUM_ITER; iter+=threads)
		{
			// Generate random (X,Y) points
			x = (double)rand_r(&seed) / (double)RAND_MAX;
			y = (double)rand_r(&seed) / (double)RAND_MAX;
			z = sqrt((x*x) + (y*y));
			// Check if point is in unit circle
			if (z <= 1.0)
				count++;
		}

		// #pragma omp critical
		// count += partial;

	}
	// Estimate Pi and display the result
	double pi = ((double)count / (double)NUM_ITER) * 4.0;
	
	printf("The result is %f\n", pi);
	printf("Took: %.5fs, threads: %d\n", omp_get_wtime()-t, omp_get_max_threads());
	
	return 0;
}

