// Based on the lecture notes from module parallelism and shared memory programming

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>

int main(){ 
	int MAX_THREADS = omp_get_max_threads();
	
	srand(0);
	int N = 1000000;
	double x[N];
		
	for(int i=0; i<N; i++){
		x[i] = ((double)(rand()) / RAND_MAX)*((double)(rand()) / RAND_MAX)*((double)(rand()) / RAND_MAX)*1000;
	}

//	double maxval = 0.0; 
//	int maxloc = 0;

	int maxloc[MAX_THREADS], mloc;
	double maxval[MAX_THREADS], mval;

	double time1;
	double time = 0;
		
	mval = 0;
	for (int j=0;j<10;j++){
			
		time1 = omp_get_wtime();
		#pragma omp parallel shared(maxval, maxloc)
		{
			int id = omp_get_thread_num();
			maxval[id]=0;

			#pragma omp for
  			for (int i=0; i < N; i++){
//		#pragma omp critical
//		{
				if (x[i] > maxval[id]){ 
					maxval[id] = x[i];	
					maxloc[id] = i;
				}
//		}
			}
		}

		for(int i = 0; i<MAX_THREADS; i++){
			if (maxval[i]>mval){
				mval = maxval[i];	
				mloc = maxloc[i];
			}	
		}

		if(j>0)
			time += omp_get_wtime()-time1;
	}
	


//	printf("Maxvalue %f at %d location \n", maxval, maxloc);
	printf("Maxvalue %f at %d location \n", mval, mloc);
	printf("With time %f \n", time/9);
}
