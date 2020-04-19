#include <stdio.h>
#include <omp.h>

#define MAX_THREADS 4

int main()
{
#pragma omp parallel
	{
		int id = omp_get_thread_num(); 
		printf("Hello World from thread %d \n", id);
	}
}
