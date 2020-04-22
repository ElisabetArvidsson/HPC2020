#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>

int main(){ 

	srand(time(0));
	int N = 100;
	double x[N];
		
	for(int i=0; i<N; i++){
		x[i] = ((double)(rand()) / RAND_MAX)*((double)(rand()) / RAND_MAX)*((double)(rand()) / RAND_MAX)*1000;
	}

	double maxval = 0.0; 
	int maxloc = 0;

	double time1 = omp_get_wtime();

  	for (int i=0; i < N; i++){
		if (x[i] > maxval){ 
        	maxval = x[i]; 
			maxloc = i;
		}
	}

//	#pragma omp parallel for
//  	for (int i=0; i < N; i++){
//		#pragma omp critical
//		{
//			if (x[i] > maxval){ 
//        		maxval = x[i]; 
//				maxloc = i;
//			}
//		}
//	}

	double time2 = omp_get_wtime();

	printf("Maxvalue %f \n", maxval);
	printf("With time %f \n", time2-time1);
}
