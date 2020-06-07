#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <math.h>


// Define constants
#define N 64
#define SEED 42

// Define precision types. Choose float or double, same for both types
typedef float ftype;
MPI_Datatype mpitype = MPI_FLOAT;


// Method declarations
void matmul(ftype* A, ftype* B, ftype* C, size_t size);
ftype sumMatrix(ftype* M, size_t size);
ftype sumMatrix(ftype* M, size_t size, MPI_Comm comm);

void printMatrix(ftype* matrix, size_t size, int rank);
void eye(ftype* M, size_t size);
void randomMatrix(ftype* M, size_t size);

void parallelMultiply(ftype* A, ftype* B, ftype* C, ftype* tmp, size_t size, 
	MPI_Comm* comm_rows, MPI_Comm* comm_cols, size_t gridsize);
void initialize(ftype* M, size_t size, size_t gridsize, int rank, 
	void(*init)(ftype* M, size_t size));


int main(int argc, char *argv[])
{

	// ================================================================================= //
	// Init MPI
	int rank, num_proc, num_threads, thread_level;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	num_threads = omp_get_max_threads();

	size_t gridsize, col, row;
	gridsize = (size_t) sqrt(num_proc);
	size_t size = N/gridsize;	// Size of each matrix block

	if(thread_level < MPI_THREAD_FUNNELED){
		printf("Insufficient level of thread support provided; %d\n", thread_level);
		return -1;
	}

	if(gridsize*gridsize != num_proc && !rank){
		printf("Number of processes (%d) is not a square number\n", num_proc);
		return -3;
	}

	if(size*gridsize != N){
		printf("Matrix size %d is not a multiple of sqrt(number of processes) %zu\n", N, gridsize);
		return -2;
	}

	if(!rank){
		printf("Initialized %d mpi processes with %d omp threads\n", num_proc, num_threads);
		printf("%d parallel processors in total\n", num_proc*num_threads);
	}

	// ================================================================================= //
	// Create MPI communicators for vertical and horizontal communication.

	col = rank % gridsize;
	row = rank/gridsize;

	MPI_Comm mpi_comm_row, mpi_comm_col;
	MPI_Comm_split(MPI_COMM_WORLD, row, rank, &mpi_comm_row);
	MPI_Comm_split(MPI_COMM_WORLD, col, rank, &mpi_comm_col);

	// ================================================================================= //
	// Allocate memory and initialize matrices. Scatter to all processes

	ftype *A, *B, *C, *tmp;

	A = (ftype*) malloc(size*size*sizeof(ftype));
	B = (ftype*) malloc(size*size*sizeof(ftype));
	C = (ftype*) malloc(size*size*sizeof(ftype));
	tmp = (ftype*) malloc(size*size*sizeof(ftype));

	srand(SEED);
	initialize(A, size, gridsize, rank, randomMatrix);
	initialize(B, size, gridsize, rank, eye);

	// ================================================================================= //
	// TODO Calculate C = AxB and measure performance

	parallelMultiply(A, B, C, tmp, size, &mpi_comm_row, &mpi_comm_col, gridsize);
	ftype sum = sumMatrix(C, size, MPI_COMM_WORLD);
	if(rank == 0){
		printf("Sum of global result: %f\n", sum);
	}

	// ================================================================================= //
	// Deallocate memory and close mpi communication

	free(A);
	free(B);
	free(C);
	free(tmp);

	MPI_Finalize();

	return 0;
}

inline size_t index(unsigned int row, unsigned int col, unsigned int stride){
	return (size_t) (row + col * stride);
}

// Generate random matrix
void randomMatrix(ftype* M, size_t size){
	unsigned int i,j;
	for (i=0; i<size; i++){
		for (j=0; j<size; j++){
			M[index(i, j, size)] = ((ftype) rand()/RAND_MAX);
		}
	}
}

// Set M to identity matrix of size size
void eye(ftype* M, size_t size){
	memset(M, 0, sizeof(ftype)*size*size);
	for(unsigned int i=0; i<size; i++){
		M[index(i,i,size)] = 1.0;
	}
}

// Scatter matrix data to other mpi processes
void scatterMatrix(ftype* send, ftype* recv, size_t size, size_t gridsize, int rank){

	// Create a MPI datatype for a matrix block
	MPI_Datatype blk_type;
	MPI_Datatype matrix_type;

	MPI_Type_vector(size, size, N, mpitype, &blk_type);
	MPI_Type_create_resized(blk_type, 0, sizeof(ftype), &matrix_type);
	MPI_Type_commit(&matrix_type);

	// Set offsets in main array for where subblocks start
	int offsets[gridsize*gridsize];
	int sendc[gridsize*gridsize];
	unsigned int o = 0;

	for(unsigned int i=0; i<gridsize; i++){
		for(unsigned int j=0; j<gridsize; j++){
			sendc[o] = 1;
			offsets[o] = (j*N*size + i*size);
			o++;
		}
	}
	
	// Send data to other processes.
	MPI_Scatterv(send, sendc, offsets, matrix_type, recv, size*size, mpitype, 0, MPI_COMM_WORLD);

}

void initialize(ftype* M, size_t size, size_t gridsize, int rank, void(*init)(ftype* M, size_t size)){

	ftype* glob;
	if(rank == 0){
		glob = (ftype*) malloc(N*N*sizeof(ftype));
		init(glob, N);
		printf("Global sum: %f\n", sumMatrix(glob, N));
	}
	scatterMatrix(glob, M, size, gridsize, rank);

	if(rank == 0)
		free(glob);

}

void printMatrix(ftype* matrix, size_t size, int rank){

	int length = size*size*8 + size +10;
	char s[length];

	sprintf(s, "Rank %02d:\n", rank);
	int offset = 9;
	
	for (int i=0; i<size; i++){
		for (int j=0; j<size; j++){
			sprintf(&s[offset], "%.05f ", matrix[index(i,j,size)]);
			offset += 8;
		}
		sprintf(&s[offset], "\n");
		offset += 1;
	} 
	std::cout << s;

}

void parallelMultiply(ftype* A, ftype* B, ftype* C, ftype* tmp, size_t size, 
	MPI_Comm* comm_rows, MPI_Comm* comm_cols, size_t gridsize){

	// Get mpi-ranks in different comm groups
	int world_rank, row_rank, col_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_rank(*comm_rows, &row_rank);
	MPI_Comm_rank(*comm_cols, &col_rank);

	// Set C to zero.
	memset(C, 0, sizeof(ftype)*size*size);

	for (unsigned int i=0; i<gridsize; i++){

		// first row has col_rank=0, second row col_rank=1 ...
		// first iteration transmits where row=col. Second where row=col+1 ...
		int root = (col_rank + i >= gridsize ? col_rank + i - gridsize : col_rank+i);

		// We are root and will broadcast A.
		// To not overwrite A of receivers, copy to tmp array
		if(root == row_rank){
			memcpy(tmp, A, sizeof(ftype)*size*size);
		}
		MPI_Bcast(tmp, size*size, mpitype, root, *comm_rows);
		
		// Multiply AxB and add to C (current A is in tmp)
		matmul(tmp, B, C, size);

		// Calculate rank of source and destination in our column group
		int dest = (col_rank - 1 < 0 ? col_rank - 1 + gridsize : col_rank - 1);
		int source = (col_rank + 1 >= gridsize ? col_rank + 1 - gridsize : col_rank + 1);

		// MPI_Sendrecv(B, size*size, mpitype, dest, i,
		// 	tmp, size*size, mpitype, source, i, *comm_cols, MPI_STATUS_IGNORE);
		// Send B to process 'above' and receive from 'below'
		MPI_Request rsend, rrecv;
		MPI_Isend(B, size*size, mpitype, dest, i, *comm_cols, &rsend);
		MPI_Irecv(tmp, size*size, mpitype, source, i, *comm_cols, &rrecv);

		// Wait for data to be sent and received
		MPI_Wait(&rsend, MPI_STATUS_IGNORE);
		MPI_Wait(&rrecv, MPI_STATUS_IGNORE);

		// Copy B from tmp array to B
		memcpy(B, tmp, sizeof(ftype)*size*size);

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

ftype sumMatrix(ftype* M, size_t size, MPI_Comm comm){

	ftype sum;
	ftype s = sumMatrix(M, size);
	MPI_Reduce(&s, &sum, 1, mpitype, MPI_SUM, 0, comm);
	
	return sum;
}

void matmul(ftype* A, ftype* B, ftype* C, size_t size){

	unsigned int i,j,k;
	for (i=0; i<size; i++)
		for(j=0; j<size; j++)
			for(k=0; k<size; k++)
				C[index(i,j,size)] += A[index(i,k,size)]*B[index(k,j,size)];

}