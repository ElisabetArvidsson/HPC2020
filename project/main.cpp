#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <math.h>


// Define constants
#define N 16
#define SEED 42

// Define precision types. Choose float or double, same for both types
typedef float ftype;
MPI_Datatype mpitype = MPI_FLOAT;


// Method declarations
void matmul(ftype* A, ftype* B, ftype* C, size_t size);
void broadcastA(ftype* A, int iteration, int row, int col, MPI_Comm comm);
void rollupB(ftype* B, MPI_Comm comm);
void initializeMatrices(ftype* A, ftype* B, ftype* C, size_t size);
ftype sumMatrix(ftype* M, size_t size);



int main(int argc, char *argv[])
{

	// ================================================================================= //
	// Init MPI
	int rank, num_proc, num_threads, thread_level;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	num_threads = omp_get_max_threads();

	if(thread_level < MPI_THREAD_FUNNELED){
		printf("Insufficient level of thread support provided; %d\n", thread_level);
		return -1;
	}

	if(N % num_proc != 0){
		printf("Matrix size %d is not a multiple of number of processes %d\n", N, num_proc);
		return -2;
	}

	if(!rank){
		printf("Initialized %d mpi processes with %d omp threads\n", num_proc, num_threads);
		printf("%d parallel processors in total\n", num_proc*num_threads);
	}

	// ================================================================================= //
	// Create MPI communicators for vertical and horizontal communication.
	int gridsize, col, row;
	gridsize = (int) sqrt(num_proc);
	if(gridsize*gridsize != num_proc && !rank){
		printf("Number of processes (%d) is not a square number. Exiting\n", num_proc);
		return -3;
	}

	col = rank % gridsize;
	row = rank/gridsize;

	MPI_Comm mpi_comm_row, mpi_comm_col;
	MPI_Comm_split(MPI_COMM_WORLD, row, rank, &mpi_comm_row);
	MPI_Comm_split(MPI_COMM_WORLD, col, rank, &mpi_comm_col);

	// ================================================================================= //
	// Allocate memory and initialize matrices

	ftype *A, *B, *C, *tmp;		// Pointers to matrices
	size_t size = N/gridsize;	// Size of each matrix block

	A = (ftype*) malloc(size*size*sizeof(ftype));
	B = (ftype*) malloc(size*size*sizeof(ftype));
	C = (ftype*) malloc(size*size*sizeof(ftype));
	tmp = (ftype*) malloc(size*size*sizeof(ftype));

	initializeMatrices(A, B, C, size);


	// ================================================================================= //
	// Calculate C = AxB and measure performance

	// TODO

	// ================================================================================= //
	// Deallocate memory and close mpi communication

	free(A);
	free(B);
	free(C);
	free(tmp);

	return 0;
}

inline unsigned int index(unsigned int row, unsigned int col, unsigned int stride){
	return row + col * stride;
}

void initializeMatrices(ftype* A, ftype* B, ftype* C, size_t size){

	srand(SEED);
	int i,j;
	for (i=0; i<size; i++){
		for (j=0; j<size; j++){
			A[index(i, j, size)] = (ftype) rand() / RAND_MAX;

		}
	}
	memset(C, 0, sizeof(ftype)*size*size);
}

void parallelMultiply(ftype* A, ftype* B, ftype* C, ftype* tmp, size_t size, 
	MPI_Comm* comm_rows, MPI_Comm* comm_cols, int gridsize){

	// Get mpi-ranks in different comm groups
	int world_rank, row_rank, col_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_rank(comm_rows, &row_rank);
	MPI_Comm_rank(comm_cols, &col_rank);

	// Set C to zero.
	memset(&C, 0, sizeof(ftype)*size*size);

	for (int i=0; i<gridsize; i++){

		// Diagonal+i broadcast A in row
		if((row_rank + i) % gridsize == col_rank){
			memcpy(&tmp, &A, sizeof(ftype)*size*size);
		}
		MPI_Bcast(&tmp, size*size, mpitype, (row_rank+i) % gridsize, mpi_comm_row);
		
		// Multiply AxB and add to C
		matmul(&tmp, &B, &C, size);

		// Send B to process 'above' and receive from 'below'
		MPI_Request rsend, rrecv;
		MPI_ISend(&B, size*size, mpitype, (col_rank-1) % gridsize, i, comm_cols, &rsend);
		MPI_IRecv(&tmp, size*size, mpitype, (col_rank+1) % gridsize, i, comm_cols, &rrecv);

		// Wait for data to be sent and received
		MPI_Wait(rsend, MPI_STATUS_IGNORE);
		MPI_Wait(rrecv, MPI_STATUS_IGNORE);

		// Copy B from tmp array to B
		memcpy(&tmp, &B, sizeof(ftype)*size*size);
			
	}
	// A,B equal to initial values, C is AxB
}

ftype sumMatrix(ftype* M, size_t size){

	ftype sum = 0;
	unsigned int i, j;
	for (i=0; i<size; i++){
		for (j=0; j<size; j++){
			sum += M[index(i, j, size)];
		}
	}
	return sum;
}

void matmul(ftype* A, ftype* B, ftype* C, size_t size){

	unsigned int i,j,k;
	for (i=0; i<size; i++)
		for(j=0; j<size; j++)
			for(k=0; k<size; k++)
				C[index(i,j,size)] += A[index(i,k,size)]*B[index(k,j,size)];

}