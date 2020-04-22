#include <stdio.h>
#include <omp.h>


int main()
{
#pragma omp parallel num_threads(6)
	{
		int id = omp_get_thread_num(); 
		printf("Hello World from thread %d \n", id);
	}
}
