#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SEED     921
#define NUM_ITER 1000000000

unsigned int flip(unsigned int iterations);
void mpi_reduce_linear(unsigned int& count, unsigned int rank, unsigned int size);
void mpi_reduce_nonblock_linear(unsigned int& count, unsigned int rank, unsigned int size);
void mpi_reduce_bintree(unsigned int& count, unsigned int rank, unsigned int size);
void mpi_reduce_bintree(unsigned int& count, unsigned int rank, unsigned int size, int level);
void mpi_reduce_gather(unsigned int& count, unsigned int rank, unsigned int size);
void mpi_reduce_reduce(unsigned int& count, unsigned int rank, unsigned int size);
void mpi_reduce_onesided(unsigned int& count, unsigned int rank, unsigned int size);

int main(int argc, char* argv[])
{

	double pi;
	int rank, processes, thread_level;
	unsigned int count;

	// function pointer to reduction function
	void (*mpi_reduce)(unsigned int&, unsigned int, unsigned int);
	// Default to linear if no argument is given
	char reduction[10];
	strncpy(reduction, "linear", 10);
	mpi_reduce = &mpi_reduce_linear;

	for (int i=0; i<argc; i++){
		if(!strcmp(argv[i], "-r") && argc > i+1){

			strncpy(reduction, argv[i+1], 10);
			if(!strcmp(argv[i+1], "linear")){
				mpi_reduce = &mpi_reduce_linear;
			}
			else if(!strcmp(argv[i+1], "bintree")){
				mpi_reduce = &mpi_reduce_bintree;
			}
			else if(!strcmp(argv[i+1], "nonblock")){
				mpi_reduce = &mpi_reduce_nonblock_linear;
			}
			else if(!strcmp(argv[i+1], "gather")){
				mpi_reduce = &mpi_reduce_gather;
			}
			else if(!strcmp(argv[i+1], "reduce")){
				mpi_reduce = &mpi_reduce_reduce;
			}
			else if(!strcmp(argv[i+1], "onesided")){
				mpi_reduce = &mpi_reduce_onesided;
			}
			else{
				printf("Unsupported reduction: %s\n", reduction);
				return -1;
			}
		}
	}


	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &thread_level);
	MPI_Comm_size(MPI_COMM_WORLD, &processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	double time = MPI_Wtime();

	srand(SEED*rank);
	count = flip(NUM_ITER/processes);

	mpi_reduce(count, rank, processes);
	
	if(rank == 0){
		// Estimate Pi and display the result
		pi = ((double)count / (double)NUM_ITER) * 4.0;
		printf("The result is %f\n", pi);
		printf("Took %.6fs using %s reduction\n", MPI_Wtime()-time, reduction);
	}
	
	MPI_Finalize();  

	return 0;
}

/**
 * Flip some coins -> Calculate pi -> ?? -> Profit!
 */
unsigned int flip(unsigned int iterations){
	unsigned int count = 0;
	double x, y, z;
	// Calculate PI following a Monte Carlo method
	for (unsigned int i = 0; i < iterations; i++)
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
	return count;
}

/**
 * Linear reduction, all send to process 0
 */
void mpi_reduce_linear(unsigned int& count, unsigned int rank, unsigned int size){

	MPI_Status status;
	unsigned int rec;
	if(rank == 0){
		for (int i=1; i<size; i++){
			MPI_Recv(&rec, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &status);
			count += rec;
		}
	}
	else{
		MPI_Send(&count, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
	}

} 

void mpi_reduce_nonblock_linear(unsigned int& count, unsigned int rank, unsigned int size){

	if(rank == 0){
		int i;
		unsigned int rec[size-1];
		MPI_Request requests[size-1];
		MPI_Status statuses[size-1];

		for(i=1; i<size; i++){
			MPI_Irecv(&rec[i-1], 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &requests[i-1]);
		}
		MPI_Waitall(size-1, requests, statuses);

		for(i=1; i<size; i++){
			count += rec[i-1];
		}
	}
	else{
		MPI_Status status;
		/* Well, technically this isn't non-blocking... 
		 * You could do MPI_Isend(...) and immediatly MPI_Wait(...), 
		 * but that just feels stupid... Since we exit after this, 
		 * we just want to stay alive until data is sent
		 */
		MPI_Send(&count, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
	}

}

/**
 * Binary reduction, send through bintree. Supports arbitrary number of processes.
 */
void mpi_reduce_bintree(unsigned int& count, unsigned int rank, unsigned int size){
	mpi_reduce_bintree(count, rank, size, 0);
}
void mpi_reduce_bintree(unsigned int& count, unsigned int rank, unsigned int size, int level){

	MPI_Status status;
	unsigned int rec;
	int target;
	int exp = 1 << level;

	// If we are higher in the bintree than its height, we are done (and we are the receiver).
	if(exp >= size){
		return;
	}
	if((rank / exp) % 2 == 0){
		// Receive data from next multiple of two up
		target = rank + exp;
		// Only receive if we have someone to receive from
		if(target < size){
			MPI_Recv(&rec, 1, MPI_UNSIGNED, target, level, MPI_COMM_WORLD, &status);
			count += rec;
		}
		// Recursively send/receive all data
		mpi_reduce_bintree(count, rank, size, level+1);
	}
	else{	
		// Send data to nex multiple of two down
		target = rank - exp;
		MPI_Send(&count, 1, MPI_UNSIGNED, target, level, MPI_COMM_WORLD);
		// Once we have sent our data, we are done.
		return;
	}
}

void mpi_reduce_gather(unsigned int& count, unsigned int rank, unsigned int size){

	unsigned int rec[size];
	MPI_Gather(&count, 1, MPI_UNSIGNED, &rec, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	
	for (int i=1; i<size; i++){
		count += rec[i];
	}

}

// I know, stupid name...
void mpi_reduce_reduce(unsigned int& count, unsigned int rank, unsigned int size){

	unsigned int rec;
	MPI_Reduce(&count, &rec, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);

	count = rec;
}

void mpi_reduce_onesided(unsigned int& count, unsigned int rank, unsigned int size){

	MPI_Win window;
	unsigned int* window_data;

	MPI_Win_allocate(size*sizeof(unsigned int), sizeof(unsigned int), MPI_INFO_NULL, MPI_COMM_WORLD, &window_data, &window);

	MPI_Win_fence(0, window);

	MPI_Put(&count, 1, MPI_UNSIGNED, 0, rank, 1, MPI_UNSIGNED, window);

	MPI_Win_fence(0, window);

	if(rank == 0){
		for (int i=1; i<size; i++){
			count += window_data[i];
		}
	}

	MPI_Win_free(&window);

}