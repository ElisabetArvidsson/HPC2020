// Based on the code from the lecture notes in module parallelism and shared memory programming 

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

	struct maxstruct {double val; int loc; char pad[128];};

	struct maxstruct maxinfo[MAX_THREADS];
	
	double time1;
	double time = 0;
		
	double mval = 0.0;
	int mloc;

	for (int j=0;j<10;j++){
			
		time1 = omp_get_wtime();
		#pragma omp parallel shared(maxinfo)
		{
			int id = omp_get_thread_num();
			maxinfo[id].val=0;
			
			#pragma omp for
  				for (int i=0; i < N; i++){
					if (x[i] > maxinfo[id].val){ 
						maxinfo[id].val = x[i];	
						maxinfo[id].loc = i;
					}
				}
		}

		for(int i = 0; i<MAX_THREADS; i++){
			if (maxinfo[i].val>mval){
				mval = maxinfo[i].val;	
				mloc = maxinfo[i].loc;
			}	
		}

		if(j>0)
			time += omp_get_wtime()-time1;
	}
	

	printf("Maxvalue %f at %d location \n", mval, mloc);
	printf("With time %f \n", time/9);
}
