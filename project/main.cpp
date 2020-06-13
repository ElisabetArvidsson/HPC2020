#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cstring>
#include <mkl.h>


// Define constants
#define BLOCK 8
#define SEED 42
#define ITERATIONS 10

#define MODE_CHECK 1
#define MODE_BENCHMARK 2

// Define precision types. Choose float or double, same for both types
typedef double ftype;
MPI_Datatype mpitype = MPI_DOUBLE;


// Method declarations
void matmul(ftype* A, ftype* B, ftype* C, size_t size);
void ompMatmul(ftype* A, ftype* B, ftype* C, size_t size);
void mklMatmul(ftype* A, ftype* B, ftype* C, size_t size);

ftype sumMatrix(ftype* M, size_t size);
ftype sumMatrix(ftype* M, size_t size, MPI_Comm comm);

void printMatrix(ftype* matrix, size_t size, int rank);
void eye(ftype* M, size_t size);
void randomMatrix(ftype* M, size_t size);

void benchmark(ftype* A, ftype* B, ftype* C, ftype* tmp, size_t size, 
	MPI_Comm* comm_rows, MPI_Comm* comm_cols, size_t gridsize, int rank);
void parallelMultiply(ftype* A, ftype* B, ftype* C, ftype* tmp, size_t size, 
	MPI_Comm* comm_rows, MPI_Comm* comm_cols, size_t gridsize);
void initialize(ftype* M, size_t size, size_t gridsize, int rank, 
	void(*init)(ftype* M, size_t size));
bool isEqual(ftype* A, ftype* B, size_t size, ftype epsilon);

void scatterMatrix(ftype* send, ftype* recv, size_t size, size_t gridsize, int rank);
void gatherMatrix(ftype* send, ftype* recv, size_t size, size_t gridsize, int rank);



int main(int argc, char *argv[])
{


	int N = 64;
	int argi = 0;

	int mode = MODE_BENCHMARK;

	while(argi < argc){
		if(!strcmp("-n", argv[argi]) && argc >= argi + 1){
			N = atoi(argv[argi+1]);
			argi++;
		}
		if(!strcmp("-c", argv[argi])){
			mode = MODE_CHECK;
		}
		else if(!strcmp("-b", argv[argi])){
			mode = MODE_BENCHMARK;
		}
		argi++;
	}

	// ====================================================================== //
	// Init MPI
	int rank, num_proc, num_threads, thread_level;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	num_threads = omp_get_max_threads();

	size_t gridsize, col, row;
	gridsize = (size_t) sqrt(num_proc);
	size_t size = N / gridsize;	// Size of each matrix block

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
		printf("%dx%d size global matrix\n", N, N);
	}

	// ====================================================================== //
	// Create MPI communicators for vertical and horizontal communication.

	col = rank % gridsize;
	row = rank / gridsize;

	MPI_Comm mpi_comm_row, mpi_comm_col;
	MPI_Comm_split(MPI_COMM_WORLD, row, rank, &mpi_comm_row);
	MPI_Comm_split(MPI_COMM_WORLD, col, rank, &mpi_comm_col);

	// ====================================================================== //
	// Allocate memory and initialize matrices. Scatter to all processes

	ftype *A0, *B0, *C0, *tmp0;
	ftype *A, *B, *C, *tmp;

	A = (ftype*) malloc(size*size*sizeof(ftype));
	B = (ftype*) malloc(size*size*sizeof(ftype));
	C = (ftype*) malloc(size*size*sizeof(ftype));
	tmp = (ftype*) malloc(size*size*sizeof(ftype));

	// ====================================================================== //
	// Calculate C = AxB and measure performance

	if(mode == MODE_BENCHMARK){
		initialize(A, size, gridsize, rank, randomMatrix);
		initialize(B, size, gridsize, rank, randomMatrix);

		benchmark(A, B, C, tmp, size, &mpi_comm_row, &mpi_comm_col, gridsize, rank);

	}

	// ====================================================================== //
	// Compare results of regular matrix multiply and parallel version

	if(mode == MODE_CHECK){

		if(!rank){
			A0 = (ftype*) malloc(N*N*sizeof(ftype));
			B0 = (ftype*) malloc(N*N*sizeof(ftype));
			C0 = (ftype*) malloc(N*N*sizeof(ftype));
			tmp0 = (ftype*) malloc(N*N*sizeof(ftype));

			srand(time(0));
			randomMatrix(A0, N);
			randomMatrix(B0, N);
		}

		scatterMatrix(A0, A, size, gridsize, rank);
		scatterMatrix(B0, B, size, gridsize, rank);

		parallelMultiply(A, B, C, tmp, size, &mpi_comm_row, &mpi_comm_col, gridsize);

		gatherMatrix(C, C0, size, gridsize, rank);

		if(!rank){
			memset(tmp0, 0, N*N*sizeof(ftype));
			ompMatmul(A0, B0, tmp0, N);

			ftype eps = (ftype) pow(10, -sizeof(ftype));	// Stricter tolerance for double
			printf("Is equal? %i\n", isEqual(C0, tmp0, N, eps));
			// ftype s1 = sumMatrix(C0, N);
			// ftype s2 = sumMatrix(tmp0, N);
			// printf("Sum C0: %f, tmp0: %f, diff: %f\n", s1, s2, s1-s2);

			free(A0);
			free(B0);
			free(C0);
			free(tmp0);
		}
	}

	// ====================================================================== //
	// Deallocate memory and close mpi communication

	free(A);
	free(B);
	free(C);
	free(tmp);

	MPI_Finalize();

	return 0;
}

// Calculate index in 1D array from 2D matrix
inline size_t index(unsigned int row, unsigned int col, unsigned int stride){
	return (size_t) (row * stride + col);
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

	MPI_Type_vector(size, size, size*gridsize, mpitype, &blk_type);
	MPI_Type_create_resized(blk_type, 0, sizeof(ftype), &matrix_type);
	MPI_Type_commit(&matrix_type);

	// Set offsets in main array for where sub-blocks start
	int offsets[gridsize*gridsize];
	int sendc[gridsize*gridsize];

	unsigned int o = 0;
	for(unsigned int i=0; i<gridsize; i++){
		for(unsigned int j=0; j<gridsize; j++){
			sendc[o] = 1;
			offsets[o] = i*size*gridsize*size + j*size;
			o++;
		}
	}
	
	// Send data to other processes.
	MPI_Scatterv(send, sendc, offsets, matrix_type, recv, size*size, mpitype, 0, MPI_COMM_WORLD);

}

// Gather matrix data from other processes
void gatherMatrix(ftype* send, ftype* recv, size_t size, size_t gridsize, int rank){

	// Create a MPI datatype for a matrix block
	MPI_Datatype blk_type;
	MPI_Datatype matrix_type;

	MPI_Type_vector(size, size, size*gridsize, mpitype, &blk_type);
	MPI_Type_create_resized(blk_type, 0, sizeof(ftype), &matrix_type);
	MPI_Type_commit(&matrix_type);

	// Set offsets in main array for where sub-blocks start
	int offsets[gridsize*gridsize];
	int recvc[gridsize*gridsize];
	unsigned int o = 0;

	for(unsigned int i=0; i<gridsize; i++){
		for(unsigned int j=0; j<gridsize; j++){
			recvc[o] = 1;
			offsets[o] = i*size*gridsize*size + j*size;
			o++;
		}
	}
	
	// Gather data from other processes.
	MPI_Gatherv(send, size*size, mpitype, recv, recvc, offsets, matrix_type, 0, MPI_COMM_WORLD);

}

// Initialize global M and scatter to all processes
void initialize(ftype* M, size_t size, size_t gridsize, int rank, void(*init)(ftype* M, size_t size)){

	ftype* glob;
	if(rank == 0){
		glob = (ftype*) malloc(size*gridsize*size*gridsize*sizeof(ftype));
		init(glob, size*gridsize);
		printf("Global sum: %f\n", sumMatrix(glob, size*gridsize));
	}
	scatterMatrix(glob, M, size, gridsize, rank);

	if(rank == 0)
		free(glob);

}

// Print all entries in M
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

// Calculte A*B=C 10 times and measure performance
void benchmark(ftype* A, ftype* B, ftype* C, ftype* tmp, size_t size, 
	MPI_Comm* comm_rows, MPI_Comm* comm_cols, size_t gridsize, int rank){

	double t1, t2;
	ftype sum;

	for (int i=0; i<ITERATIONS; i++){
		t1 = MPI_Wtime();
		parallelMultiply(A, B, C, tmp, size, comm_rows, comm_cols, gridsize);
		t2 += MPI_Wtime()-t1;

		sum = sumMatrix(C, size, MPI_COMM_WORLD);
		if(rank == 0){
			printf("Time: %f, sum: %f\n", MPI_Wtime()-t1, sum);
		}
	}
	if(rank == 0)
		printf("Avg time: %f\n", t2/(double)ITERATIONS);

}

// Calculate A*B=C concurrently on all processes
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
		//ompMatmul(tmp, B, C, size);
		mklMatmul(tmp, B, C, size); 	// switch between matmul/ompMatmul/mklMatmul for different versions

		// Calculate rank of source and destination in our column group
		int dest = (col_rank - 1 < 0 ? col_rank - 1 + gridsize : col_rank - 1);
		int source = (col_rank + 1 >= gridsize ? col_rank + 1 - gridsize : col_rank + 1);

		// Send B to process 'above' and receive from 'below'
		// MPI_Sendrecv(B, size*size, mpitype, dest, i,
		// 	tmp, size*size, mpitype, source, i, *comm_cols, MPI_STATUS_IGNORE);
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

// Calculate A*B=C
void matmul(ftype* A, ftype* B, ftype* C, size_t size){

	unsigned int i,j,k;
	for (i=0; i<size; i++)
		for(k=0; k<size; k++)
			for(j=0; j<size; j++)
				C[index(i,j,size)] += A[index(i,k,size)]*B[index(k,j,size)];

}

// Calculate A*B=C with openmp
void ompMatmul(ftype* A, ftype* B, ftype* C, size_t size){

	unsigned int i,j,k;
	// unsigned int ii,jj,kk;
	#pragma omp parallel for private(i,j,k) schedule(static)
	for (i = 0; i < size; i++)
		for(k = 0; k < size; k++)
			for(j = 0; j < size; j++)
				C[index(i,j,size)] += A[index(i,k,size)]*B[index(k,j,size)];

}

// Calculate A*B=C with mkl
void mklMatmul(ftype* A, ftype* B, ftype* C, size_t size){

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			size, size, size, 1.0,
			A, size,
			B, size,
		       	1.0,	// Add to C each iteration
			C, size
		   );

}

// Check if A_ij == B_ij up to some epsilon
bool isEqual(ftype* A, ftype* B, size_t size, ftype epsilon){

	unsigned int i,j;
	for(i = 0; i < size; i++)
		for(j = 0; j < size; j++)
			if(fabs(A[index(i, j, size)] / B[index(i, j, size)] -1) > epsilon){
				printf("A:%f, B:%f\n", A[index(i, j, size)], B[index(i, j, size)]);
				return false;
			}
			
	return true;

}
