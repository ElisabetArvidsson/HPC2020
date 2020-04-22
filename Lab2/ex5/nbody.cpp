#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <iostream>

#define DIM 3

typedef double fp_t;
typedef fp_t vect_t[DIM];

int num_particles = 100;
int iterations = 100;
bool verbose = false;
bool reduced = false;

const int x = 0;
const int y = 1;

const fp_t G = 0.01;
const fp_t dt = 0.05;


void initialize(vect_t *pos, vect_t *vel, fp_t *mass){

	// Predictability
	srand(1);

	// Don't parallelize, it would destroy rand() predictability and initialize differently
	for(int i=0; i<num_particles; i++){

		#pragma unroll
		for (int d=0; d<DIM; d++){
			pos[i][d] = (rand() / (double)(RAND_MAX)) * 2 - 1;
			vel[i][d] = (rand() / (double)(RAND_MAX)) * 2 - 1;
		}
		mass[i] = (rand() / (double)(RAND_MAX));

	}
}

void printStatus(vect_t *pos, vect_t *vel, float T){

	printf("Time %.2f\n", T);
	for(int i=0; i<num_particles; i++){
		printf("%04d, pos: (%.8f,%.8f), vel: (%.8f,%.8f)\n", i, pos[i][x],pos[i][y], vel[i][x], vel[i][y]);
	}
}


/**
 * Calculate the force between two particles.
 */
inline void getForce(fp_t *f, vect_t *pos, fp_t *mass, int q, int k){
	fp_t l, inv_dist;
	register fp_t dist = 0;

	#pragma unroll
	for(int d=0; d<DIM; d++){
		l = pos[q][d]-pos[k][d];
		dist += (l*l);
	}

	dist = sqrt(dist);
	inv_dist = 1/(dist*dist*dist);

	#pragma unroll
	for(int d=0; d<DIM; d++){
		f[d] = G * mass[q] * mass[k] * inv_dist * (pos[q][d]-pos[k][d]);
	}
}

/**
 * Step the simulation forward one step. Calculate forces individually between all particles
 */ 
void step(vect_t *pos, vect_t *vel, fp_t *mass, vect_t *force){

	#pragma omp parallel for schedule(guided)
	for(int i=0; i<num_particles; i++){

		for(int j=0; j<num_particles; j++){

			if(i==j){
				continue;
			}

			fp_t* f = new fp_t[DIM];
			memset(f, 0, DIM * sizeof(fp_t));
			getForce(f, pos, mass, i, j);

			#pragma unroll
			for(int d=0; d<DIM; d++){
				force[i][d] += f[d];
			}
			delete[] f;
		}
	}
}

/**
 * Step the simulation forward one step. Reduced algorithm that calculated 
 * the forces once for each pair of particles
 */ 
void step_reduced(vect_t *pos, vect_t *vel, fp_t *mass, vect_t *force){

	#pragma omp parallel for schedule(guided)
	for(int i=0; i<num_particles; i++){

		for(int j=0; j<i; j++){

			fp_t* f = new fp_t[DIM];
			memset(f, 0, DIM * sizeof(fp_t));
			getForce(f, pos, mass, i, j);

			#pragma unroll
			for(int d=0; d<DIM; d++){

				// atmoics here to avoid race condition
				#pragma omp atomic
				force[i][d] += f[d];
				#pragma omp atomic
				force[j][d] -= f[d];
				
			}
			delete[] f;

		}
	}
}

/**
 * Update position and velocities of all particles from their current accelleration
 */
void move(vect_t *pos, vect_t *vel, fp_t *mass, vect_t *force){
	// Update position/velocity
	#pragma omp parallel for schedule(guided)
	for(int i=0; i<num_particles; i++){
		#pragma unroll
		for(int d=0; d<DIM; d++){
			pos[i][d] += dt*vel[i][d];
			vel[i][d] += dt*force[i][d] / mass[i];
		}	
	}
}

/**
 * Set up and run the simulation
 */
int main(int argc, char const *argv[])
{


	int i=1;
	while(i<argc){

		if(!strcmp(argv[i], "-n")){
			num_particles = atoi(argv[i+1]);
			i++;
		}
		else if(!strcmp(argv[i], "-i")){
			iterations = atoi(argv[i+1]);
			i++;
		}
		else if(!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")){
			verbose = true;
		}
		else if(!strcmp(argv[i], "-r") || !strcmp(argv[i], "--reduced")){
			reduced = true;
		}
		else {
			printf("Ignoring unknown argument %s\n", argv[i]);
		}
		i++;
	}

	printf("Number of particles: %d, iterations: %d\n", num_particles, iterations);
	if(reduced)
		std::cout << "Using reduced iterator" << std::endl;
	else
		std::cout << "Using naive iterator" << std::endl;
	printf("Using %d threads in parallel\n", omp_get_max_threads());


	// Allocate memory for 
	vect_t* pos = (vect_t*) malloc(num_particles * sizeof(vect_t));
	vect_t* vel = (vect_t*) malloc(num_particles * sizeof(vect_t));
	fp_t* mass = (fp_t*) malloc(num_particles * sizeof(fp_t));
	vect_t* force = (vect_t*) malloc(num_particles * sizeof(vect_t));


	double start, end;
	double time = 0;
	double timesq = 0;
	float T = 0;
	int intervall = 10;

	// Initialize particle system
	initialize(pos, vel, mass);
	if(verbose)
		printStatus(pos, vel, T);

	// Run simulation
	for (int j=0; j<iterations; j+=intervall){

		start = omp_get_wtime();
		for (int i=0; i<intervall; i++){

			// Set forces to zero
			memset(force, 0, num_particles * sizeof(vect_t));
			// Step simulation forward
			if(reduced)
				step_reduced(pos, vel, mass, force);
			else
				step(pos, vel, mass, force);

			move(pos, vel, mass, force);
			T += dt;

		}
		end = omp_get_wtime();
		time += (end-start);
		timesq += (end-start)*(end-start);

		// Print current simulation status
		if(verbose)
			printStatus(pos, vel, T);
	}

	// Free memory allocation
	free(pos);
	free(vel);
	free(mass);
	free(force);

	// Output performance figures
	double var = (timesq - time*time / iterations*intervall)/(iterations / intervall -1 );
	printf("Total time: %.3f\n", time);
	printf("Average time per iteration: %.5f +- %.7f\n", time/iterations, var);

	return 0;
}
