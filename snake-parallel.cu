//%%cuda --name snake-super.cu
/*	CS6023 GPU Programming
 	Project - Genetic Algorithm to optimise snakes game
 		Done By, 
 		Shivam Mittal, cs16b038
        Rachit Tibrewal, cs16b022
        R Sai Harshini, cs16b112
 	Parallel Code
*/

#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

using namespace std;

// utility functio used for debugging the code.
#define cudaErrorTrace() {\
    cudaError_t err = cudaGetLastError();\
    if(err != cudaSuccess) {\
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);\
    }\
}

typedef pair<int, int> ii;

#define POPULATION_SIZE 4096
#define NUM_GENERATIONS 300
#define NUM_FOODS 256

// Defining the number parameters in each layer of the neural network. 
// The neural network sed consists of three layers, the input layer, one hidden layer and the output layer. Sigmod activationg is used at each step
// n is the number of parameter in the input layer.  m1 is the number of parameter in the hidden layer. o is the number of parameter in the output layer.
#define n 24
#define m1 16
#define o 4

// For random number generation.
std::random_device oracle{};
auto tempo = oracle();    
mt19937 rd{tempo};

int GENOME_LENGTH;

int *fitness_score = NULL, max_score;

// M and N are used to define the initial x and y coordinate of the snake respectively.
#define M 80
#define N 80
// Q_len gives the upperbound on the length of the snake.
#define Q_LEN 128

/* Device funcion. This is a utility function which is called from the evaluate kernel. This function is used to check (i,j) lies on the line joining (0,0) 
and (u,v) (same direction wrt origin)*/ 
__device__
bool check(int u, int v, int i, int j) {
	if(u == 0 && v != 0) {
		return (i == 0) & (j == v / abs(v));
	} else if(u != 0 && v == 0) {
		return (i == u / abs(u)) & (j == 0);
	} else if(u != 0 && v != 0) {
		return (i == u / abs(u)) & (j == v / abs(v));
	}
	return false;
}

/* Device funcion. This is a utility function which is called from the evaluate kernel. It calculates one output of a layer of the neural network (outpout[idx]) */
__device__
float dense(float layerInput[],float W[], float b[],int idx,int in_len,int out_len)
{
	float output = 0.0;
	for(int j = 0; j < in_len; j++) {
		output += W[j * out_len + idx] * layerInput[j];
	}
	output += b[idx];
	/* sigmoid */
	output = 1.0 / (1.0 + expf(-output));
	return output;
}

/* Device code. Kerenl to evaluate the fitness score of all the organisms in a generation parallely One bblock is given to one organism 
and threads in the block are further utilised to divide work among the organism for the neural evaluation part. */
__global__
void evaluate(float *genes, int *foods, int *fitness_score, int GENOME_LENGTH) {
	// Variables for score computation
	int food_pos[2];
	int snake[Q_LEN][2];
	int snake_init_length;
	int st;
	int en;
	int maxiters;
	int additers;
	int loops;
	int snake_motion;
	int dir[2];
	int score;
	int fi;
	__shared__ float dist[8][3];
	float *input;
	float *W1;
	float *b1;
	float *W2;
	float *b2;
	__shared__ float output1[m1];
	__shared__ float output2[o];
	__shared__ int com;
	int head[2];
	int snake_size;
	int x, y;
	int snakeIsAlive;
	int foodEaten;
	
	extern __shared__ float gene[];

	int i = threadIdx.x;

	// each gene copied into shared memory by each thread.
	while(i < GENOME_LENGTH) {
		gene[i] = genes[blockIdx.x * GENOME_LENGTH + i];
		i += blockDim.x;
	}

	__syncthreads();
	
	// work done by all the threads.
	int init_x = M / 2;
	int init_y = N / 2;

	input = &dist[0][0];
	W1 = &gene[0];
	b1 = &gene[n * m1];
	W2 = &gene[n * m1 + m1];
	b2 = &gene[n * m1 + m1 + m1 * o];
	snake_init_length = 5;
	st = 0;
	en = snake_init_length;
	maxiters = 3 * (M + N);
	additers = 1 * (M + N);
	loops = maxiters;
	score = 0;
	fi = 0;
	snake_motion = 3;

	// directions.
	// 1 north:0,-1
	// 2 south:0,1
	// 3 east: 1,0
	// 4 west:-1,0

	dir[0] = 1;
	dir[1] = 0;
	snakeIsAlive = 1;
	foodEaten = 1;

	for(int i = 0; i < snake_init_length; i++) {
		snake[i][0] = i + init_x;
		snake[i][1] = init_y;
	}

	// loop till the anske is alive
	do
	{
		if(foodEaten) {
			food_pos[0] = foods[2 * fi];
			food_pos[1] = foods[2 * fi + 1];
			fi++;
			foodEaten = 0; 
		}

		// calculating the input patrameters of the neural network
		head[0] = snake[(en - 1 + Q_LEN) % Q_LEN][0];
		head[1] = snake[(en - 1 + Q_LEN) % Q_LEN][1]; 
		x = head[0];
		y = head[1];
		snake_size = (en - st + Q_LEN) % Q_LEN;

		if(threadIdx.x < 24) {
			int i = threadIdx.x / 3;
			int j = threadIdx.x % 3;
			dist[i][j] = 2 * (M + N);
		}

		__syncthreads();

		if (threadIdx.x < 9 && threadIdx.x != 4) {
			int i = threadIdx.x / 3 - 1;
			int j = threadIdx.x % 3 - 1;
			int k = (threadIdx.x > 4) ? (threadIdx.x - 1) : threadIdx.x;
			
			if(i == 0) {
				dist[k][0] = (j > 0) * N - j * y;
			} else if(j == 0) {
				dist[k][0] = (i > 0) * M - i * x;
			} else {
				dist[k][0] = min((i > 0) * M - i * x, (j > 0) * N - j * y);
			}

			int u, v;
			u = food_pos[0] - x;
			v = food_pos[1] - y;
			if(check(u, v, i, j)) {
				if(abs(dist[k][1]) > float(abs(u) + abs(v)) / (abs(i) + abs(j))) {
					dist[k][1] = float(abs(u) + abs(v)) / (abs(i) + abs(j));
				}
			}

			// TODO
			for(int ti = 0; ti < snake_size; ti++) {
				int haha[2];
				haha[0] = snake[st][0];
				haha[1] = snake[st][1];
				// snake.pop();
				st = (st + 1 + Q_LEN) % Q_LEN;
				u = haha[0] - x;
				v = haha[1] - y;
				if(check(u, v, i, j)) {
					if(abs(dist[k][2]) > float(abs(u) + abs(v))/(abs(i) + abs(j))) {
						dist[k][2] = float(abs(u) + abs(v))/(abs(i) + abs(j));
					}
				}
				snake[en][0] = haha[0];
				snake[en][1] = haha[1];
				en = (en + 1 + Q_LEN) % Q_LEN;
				// snake.push(haha);
			}
		}

		__syncthreads();
		
		// Evauating the two layers of the neural network.
		// First layer
		if(threadIdx.x < m1) {
			int i = threadIdx.x;
			/* dense 1  */
			output1[i] = dense(input,W1,b1,i,n,m1);
		}
		
		__syncthreads();
		
		// Second layer
		if (threadIdx.x < o) {
			int i = threadIdx.x;
			/* dense 2  */
			output2[i] = dense(output1,W2,b2,i,m1,o);
		}

		__syncthreads();
		
		// getting the combined vale of the neural network.
		if(threadIdx.x == 0) {
			float maxm = output2[0];
			com = 0;
			for(int i = 1; i < o; i++) {
				if (output2[i] > maxm) {
					maxm = output2[i];
					com = i;
				}
			}	
		}
		
		__syncthreads();
		
		//Updating the directions of the snake.
		/** com = 
			 * 0 straight
			 * 1 left
			 * 2 right
			 * 3 backward
			 */

		if(com == 1) {
			if(snake_motion == 1) {
				// change to west
				snake_motion = 4;
				dir[0] = -1;
				dir[1] = 0;
			} else if(snake_motion == 2) {
				// change to east
				snake_motion = 3;
				// dir = ii(1, 1.0f);
				dir[0] = 1;
				dir[1] = 0;
			} else if(snake_motion == 3) {
				// change to north
				snake_motion = 1;
				// dir = ii(0, -1);
				dir[0] = 0;
				dir[1] = -1;
			} else if(snake_motion == 4) {
				// change to south
				snake_motion = 2;
				// dir = ii(0, 1);
				dir[0] = 0;
				dir[1] = 1;
			}
		} else if(com == 2){
			if(snake_motion == 1) {
				// change to east
				snake_motion = 3;
				// dir = ii(1, 0);
				dir[0] = 1;
				dir[1] = 0;
			} else if(snake_motion == 2) {
				// change to west
				snake_motion = 4;
				// dir = ii(-1, 0);
				dir[0] = -1;
				dir[1] = 0;
			} else if(snake_motion == 3) {
				// change to south
				snake_motion = 2;
				// dir = ii(0, 1);
				dir[0] = 0;
				dir[1] = 1;
			} else if(snake_motion == 4) {
				// change to north
				snake_motion = 1;
				// dir = ii(0, -1);
				dir[0] = 0;
				dir[1] = -1;
			}
		} else if(com == 3) {
			snakeIsAlive = 0;
			// break;
		}
		
		
		// move the snake in the direction
		head[0] = head[0] + dir[0];
		head[1] = head[1] + dir[1];
		snake[en][0] = head[0];
		snake[en][1] = head[1];
		en = (en + 1 + Q_LEN) % Q_LEN;

		// check if the snake eats the food in the next move
		if(head[0] != food_pos[0] || head[1] != food_pos[1]) {
			st = (st + 1 + Q_LEN) % Q_LEN;
		} else {
			score += 1;
			loops += additers;
			foodEaten = 1;
		}

		// check if the snake crosses any boundaries
		x = head[0];
		y = head[1];
		if(x < 0 || y < 0 || x >= M || y >= N) {
			// crossed the boundart game over
			snakeIsAlive = 0;
			// break;
		}

		// TODO
		// check if the snake eats it self
		snake_size = (en - st + Q_LEN) % Q_LEN;
		for(int i = 0; i < snake_size; i++) {
			int haha[2];
			haha[0] = snake[st][0];
			haha[1] = snake[st][1];
			// snake.pop();
			st = (st + 1 + Q_LEN) % Q_LEN;
			if(i < snake_size - 1 && haha[0] == x && haha[1] == y) {
				snakeIsAlive = 0;
				break;
			}    
			snake[en][0] = haha[0];
			snake[en][1] = haha[1];
			en = (en + 1 + Q_LEN) % Q_LEN;        
			// snake.push(haha);
		}		
		loops--;
	} while(snakeIsAlive && loops >= 0 && fi < NUM_FOODS);
	
	__syncthreads();

	// Updating the fitness score.
	if(threadIdx.x == 0) {
		fitness_score[blockIdx.x] = score;
	}
}

// Function to select the best (selection_cutoff)% of the population in each generation where the organisms are sorted in the decreasing of the fitness scores.
__global__
void selection(float *prev, float *curr, int *idx) {
	// Using ping-pong method here.
	curr[blockIdx.x * blockDim.x + threadIdx.x] = prev[idx[blockIdx.x] * blockDim.x + threadIdx.x];
}

// Function to crossover between the the best population selected by the selection function and create the next generation.
// The next generation comprises of the best population selected in the current generation and the organisms generated by their crossover.
__global__
void crossover(unsigned int *rand1, unsigned int *rand2, float *d_organism, const int offset) {
	int idx = blockIdx.x;
	
	if(threadIdx.x <= rand2[idx] % blockDim.x) {
		d_organism[(offset + blockIdx.x) * blockDim.x + threadIdx.x] = d_organism[(rand1[idx] % offset) * blockDim.x + threadIdx.x];
	} else {
		d_organism[(offset + blockIdx.x) * blockDim.x + threadIdx.x] = d_organism[(rand1[blockDim.x + idx] % offset) * blockDim.x + threadIdx.x];
	}
}

// Function to mutate the genomes of each organism. Mutation is one of the fundamental concept of genetic algorithms. 
__global__
void mutate(float *rand1, float *rand2, float *d_organism, const float mutation_rate) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	float changed = d_organism[idx];

	if(rand1[idx] < mutation_rate) {
		changed += rand2[idx] / 5.0;
 		changed = max(-1.0f, changed);
 		changed = min(1.0f, changed);
	}

	d_organism[idx] = changed;
}

/* kernel to adjust the range of uniform value to (-1, 1] */
__global__ 
void scale(float *mat, float a, float b) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	mat[idx] = (mat[idx] - 0.5) * (b - a) + (b + a) / 2.0;
}

int main() {
	srand(time(NULL));

    GENOME_LENGTH = n * m1 + m1 + m1 * o + o;
	
	printf("Genome length: %d\n", GENOME_LENGTH);
	printf("Generation size: %d\n", POPULATION_SIZE);

	const int L = POPULATION_SIZE * GENOME_LENGTH;
	const size_t data_size = sizeof(float) * POPULATION_SIZE * GENOME_LENGTH;
	
	float *best_organism = (float *) malloc(sizeof(float) * GENOME_LENGTH);
	
	int blocks = 4096;
    int threads = GENOME_LENGTH;
	
	float *d_organism, *d_temp_generation;

	/* Allocate memory for organisms on device */
	cudaMalloc((void**) &d_organism, data_size);
	cudaMalloc((void**) &d_temp_generation, data_size);
	cudaErrorTrace();

	curandGenerator_t prng;
	
	/* Create pseudo random number generator */
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_MT19937);
	cudaErrorTrace();

	/* Set seed */
	curandSetPseudoRandomGeneratorSeed(prng, 42ULL);
	cudaErrorTrace();
	
	/** 
	* Create genomes by uniform initialization of organism matrix
	*/
	curandGenerateUniform(prng, d_organism, L);
	cudaErrorTrace();	
	
	/* adjust the range of uniform value to (-1, 1] */
	scale<<<POPULATION_SIZE, GENOME_LENGTH>>>(d_organism, -1.0, 1.0);

	max_score = 0;

	FILE *fout = fopen("genomes.txt", "w");

	fprintf(fout, "NUM_GENERATIONS = %d\n", NUM_GENERATIONS);
	fprintf(fout, "POPULATION_SIZE = %d\n", POPULATION_SIZE);
	fprintf(fout, "GENOME_LENGTH = %d\n", GENOME_LENGTH);

	int *d_fitness_score, *d_indices;

	/* Allocate memory for fitness score on host */
	fitness_score = (int *) malloc(sizeof(int) * POPULATION_SIZE);

	/* Allocate memory for fitness score on device */
	cudaMalloc((void**) &d_fitness_score, sizeof(int) * POPULATION_SIZE);
	cudaMalloc((void**) &d_indices, sizeof(int) * POPULATION_SIZE);
	cudaErrorTrace();

	thrust::device_ptr<int> thrust_indices_ptr(d_indices);
	thrust::device_ptr<int> thrust_fitness_score_ptr(d_fitness_score);
	
	const size_t food_size = sizeof(int) * 2 * NUM_FOODS;
	
	int *h_foods, *d_foods;
	
	h_foods = (int *) malloc(food_size);
	cudaMalloc((void**) &d_foods, food_size);
	cudaErrorTrace();
	
	unsigned int *random_uints[2];
	float *random_floats[2];
	
	cudaMalloc((void**) &random_uints[0], sizeof(int) * 2 * POPULATION_SIZE);
	cudaMalloc((void**) &random_uints[1], sizeof(int) * POPULATION_SIZE);
	cudaMalloc((void**) &random_floats[0], sizeof(int) * POPULATION_SIZE * GENOME_LENGTH);
	cudaMalloc((void**) &random_floats[1], sizeof(int) * POPULATION_SIZE * GENOME_LENGTH);


	cudaStream_t stream1;
  	cudaStreamCreate(&stream1);
	for(int i = 0; i < NUM_GENERATIONS; i++) {
		int local_max = -1, local_best;

		for(int k = 0; k < NUM_FOODS; k++) {
			h_foods[2 * k] = rand() % M;
			h_foods[2 * k + 1] = rand() % N;
		}

		/* Copy food positions from host to device */
		cudaMemcpy(d_foods, h_foods, food_size, cudaMemcpyHostToDevice);
		cudaErrorTrace();
		
		/**
		Logic:
		Each organism is allocated a block,
		the threads in the blocks perform
		the operations for the organism 
		*/
		blocks = POPULATION_SIZE;
		threads = 32;

		evaluate<<<blocks, threads, sizeof(float) * GENOME_LENGTH>>>(d_organism, d_foods, d_fitness_score, GENOME_LENGTH);

		curandSetStream(prng,stream1);
		curandGenerate(prng, random_uints[0], 2 * POPULATION_SIZE);
		curandGenerate(prng, random_uints[1], POPULATION_SIZE);	
		curandGenerateUniform(prng, random_floats[0], POPULATION_SIZE * GENOME_LENGTH);
		curandGenerateNormal(prng, random_floats[1], POPULATION_SIZE * GENOME_LENGTH, 0.0, 1.0);
		thrust::sequence(thrust::cuda::par.on(stream1), thrust_indices_ptr, thrust_indices_ptr + POPULATION_SIZE);
		

		cudaMemcpy(fitness_score, d_fitness_score, sizeof(int) * POPULATION_SIZE, cudaMemcpyDeviceToHost);
		cudaErrorTrace();
		
		for(int j = 0; j < POPULATION_SIZE; j++) {
			if(local_max < fitness_score[j]) {
				local_max = fitness_score[j];
				local_best = j;
			}
		}

		cudaMemcpy(best_organism, d_organism + local_best * GENOME_LENGTH, sizeof(float) * GENOME_LENGTH, cudaMemcpyDeviceToHost);
		cudaErrorTrace();

		// printing the genome of the best organism in the generation to the file.
		for(int k = 0; k < GENOME_LENGTH; k++) {
			fprintf(fout, "%f ", best_organism[k]);
		}
		fprintf(fout, "\n");

		max_score = max(max_score, local_max);

		printf("Score after generation %d => local: %d | max: %d\n", i, local_max, max_score);

		const int selected = 0.15 * POPULATION_SIZE;

		
		cudaStreamSynchronize(stream1);	
		thrust::sort_by_key(thrust_fitness_score_ptr, thrust_fitness_score_ptr + POPULATION_SIZE, thrust_indices_ptr, thrust::greater<int>());
		selection<<<selected, GENOME_LENGTH>>>(d_organism, d_temp_generation, d_indices);

		float *temp_ptr = d_temp_generation;
		d_temp_generation = d_organism;
		d_organism = temp_ptr;
				
		
		crossover<<<POPULATION_SIZE - selected, GENOME_LENGTH>>>(random_uints[0], random_uints[1], d_organism, selected);
		mutate<<<POPULATION_SIZE, GENOME_LENGTH>>>(random_floats[0], random_floats[1], d_organism, 1e-2);
	}

	fclose(fout);

	cudaStreamDestroy(stream1);
	cudaFree(d_organism);
	cudaFree(d_temp_generation);
	cudaFree(d_fitness_score);
	cudaFree(d_indices);
	cudaFree(d_foods);
	cudaFree(random_uints[0]);
	cudaFree(random_uints[1]);
	cudaFree(random_floats[0]);
	cudaFree(random_floats[1]);

	free(best_organism);

	return 0;
}