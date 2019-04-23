%%cuda --name snake-super.cu
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

using namespace std;

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

// Defining the parameters in each layer of the neural network.
// int n = 24, m1 = 16, o = 4;
#define n 24
#define m1 16
#define o 4

std::random_device oracle{};
auto tempo = oracle();    
mt19937 rd{tempo};

int GENOME_LENGTH;

float *organism;

int *fitness_score = NULL, max_score;

#define M 80
#define N 80
#define Q_LEN 128

__device__
bool check(int u, int v, int i, int j) {
	if(u == 0 && v != 0) {
		if(i == u && j == v / abs(v)) {
			return true;
		}
	}
	else if(u != 0 && v == 0) {
		if(i == u / abs(u) && j == v) {
			return true;
		}
	}
	else if(u != 0 && v != 0) {
		if(i == u / abs(u) && j == v / abs(v)) {
			return true;
		}
	}
	return false;
}

__global__
void evaluate(float* genes, int* foods, int* fitness_score, int GENOME_LENGTH) {
	__shared__ int food_pos[2];
	__shared__ int snake[Q_LEN][2];
	__shared__ int snake_init_length;
	__shared__ int st;
	__shared__ int en;
	__shared__ int maxiters;
	__shared__ int additers;
	__shared__ int loops;
	__shared__ int snake_motion;
	__shared__ int dir[2];
	__shared__ int score;
	__shared__ int fi;
	__shared__ float dist[8][3];
	__shared__ float* input;
	__shared__ float* W1;
	__shared__ float* b1;
	__shared__ float* W2;
	__shared__ float* b2;
	__shared__ float output1[m1];
	__shared__ float output2[o];
	__shared__ int head[2];
	__shared__ int snake_size;
	__shared__ int x, y;
	__shared__ int snakeIsAlive;
	__shared__ int foodEaten;
	
	extern __shared__ float gene[];

	int i = threadIdx.x;

	while(i < GENOME_LENGTH) {
		gene[i] = genes[blockIdx.x * GENOME_LENGTH + i];
		i += blockDim.x;
	}

	int init_x = M / 2;
	int init_y = N / 2;

	if(threadIdx.x == 0) {
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
		dir[0] = 1;
		dir[1] = 0;
		snakeIsAlive = 1;
		foodEaten = 1;
	}

	__syncthreads();
	
	if(threadIdx.x < snake_init_length) {
		snake[threadIdx.x][0] = threadIdx.x + init_x;
		snake[threadIdx.x][1] = init_y; 
	}
		
		if(threadIdx.x == 0) {
	do
	{
		// __syncthreads();
		// if(threadIdx.x == 0) {
			if(foodEaten) {
				food_pos[0] = foods[2 * fi];
				food_pos[1] = foods[2 * fi + 1];
				fi++;
				foodEaten = 0; 
			}

			head[0] = snake[(en - 1 + Q_LEN) % Q_LEN][0];
			head[1] = snake[(en - 1 + Q_LEN) % Q_LEN][1]; 
			x = head[0];
			y = head[1];
			snake_size = (en - st + Q_LEN) % Q_LEN;
			for(int i = 0; i < 8; i++) {
				for(int j = 0; j < 3; j++) {
					dist[i][j] = 2 * max(M, N);
				}
			}
			int k = 0;
			for(int i = -1; i <= 1; i++) {
				for(int j = -1; j <= 1; j++) {
					if (i == 0 && j == 0 ) {
						continue;
					} else if(i == 0) {
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
					k++;						
				}
			}

			/* dense 1 */
			for(int i = 0; i < m1; i++) {
				output1[i] = 0;
				for(int j = 0; j < n; j++) {
					output1[i] += W1[j * m1 + i] * input[j];
				}
				output1[i] += b1[i];
			}
			/* sigmoid */
			for(int i = 0; i < m1; i++) {
				output1[i] = 1.0 / (1.0 + expf(-output1[i]));
			}

			/* dense 2 */
			for(int i = 0; i < o; i++) {
				output2[i] = 0;
				for(int j = 0; j < m1; j++) {
					output2[i] += W2[j * o + i] * output1[j];
				}
				output2[i] += b2[i];
			}

			/* sigmoid */
			for(int i = 0; i < m1; i++) {
				output2[i] = 1.0 / (1.0 + expf(-output2[i]));
			}
			float maxm = output2[0];
			int com = 0;
			for(int i = 1; i < o; i++) {
				if (output2[i] > maxm) {
					maxm = output2[i];
					com = i;
				}
			}	

			if(com == 0) {
				// no change to direction
			} else if(com == 1) {
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
			
			// check if the snake eats the food in the next move
			// head = ii(head.first + dir.first, head.second + dir.second); 
			head[0] = head[0] + dir[0];
			head[1] = head[1] + dir[1];
			snake[en][0] = head[0];
			snake[en][1] = head[1];
			en = (en + 1 + Q_LEN) % Q_LEN;

			// move the snake in the direction
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
			
			if(!snakeIsAlive) {
				//snake is not alive
				// break;
			}
			
			loops--;
		// }
		// __syncthreads();
	} while(snakeIsAlive && loops >= 0 && fi < NUM_FOODS);
		}
	
	__syncthreads();

	if(threadIdx.x == 0) {
		fitness_score[blockIdx.x] = score;
	}
}

// Function to select the best (selection_cutoff)% of the population in each generation where the organisms are sorted in the decreasing of the fitness scores.
int selection(float selection_cutoff) {
	int selected = 0;
	float *new_generation = (float *) malloc(sizeof(float) * POPULATION_SIZE * GENOME_LENGTH);
	
	ii list[POPULATION_SIZE];
	
	for(int i = 0; i < POPULATION_SIZE; i++) {
		list[i] = {-fitness_score[i], i};
	}

	sort(list, list + POPULATION_SIZE);

	for(int i = 0; i < int(POPULATION_SIZE * selection_cutoff); i++) {
		copy(organism + list[i].second * GENOME_LENGTH, organism + (list[i].second + 1) * GENOME_LENGTH, new_generation + selected * GENOME_LENGTH);
		selected++;
	}

	free(organism);
	organism = new_generation;
	return selected;
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

__global__ 
void scale(float* mat, float a, float b) {
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
	
	organism = (float *) malloc(data_size);
	
	int blocks = 4096;
    int threads = GENOME_LENGTH;
	
	float* d_organism;

	/* Allocate memory for organisms on device */
	cudaMalloc((void**) &d_organism, data_size);
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
	scale<<<blocks, threads>>>(d_organism, -1.0, 1.0);

	/* Copy device memory to host */
	cudaMemcpy(organism, d_organism, data_size, cudaMemcpyDeviceToHost);
	cudaErrorTrace();

	max_score = 0;

	FILE *fout = fopen("genomes.txt", "w");

	fprintf(fout, "NUM_GENERATIONS = %d\n", NUM_GENERATIONS);
	fprintf(fout, "POPULATION_SIZE = %d\n", POPULATION_SIZE);
	fprintf(fout, "GENOME_LENGTH = %d\n", GENOME_LENGTH);

	int* d_fitness_score;

	/* Allocate memory for fitness score on host */
	fitness_score = (int*) malloc(sizeof(int) * POPULATION_SIZE);

	/* Allocate memory for fitness score on device */
	cudaMalloc((void**) &d_fitness_score, sizeof(int) * POPULATION_SIZE);
	cudaErrorTrace();
	
	const size_t food_size = sizeof(int) * 2 * NUM_FOODS;
	
	int *h_foods, *d_foods;
	
	h_foods = (int*) malloc(food_size);
	cudaMalloc((void**) &d_foods, food_size);
	cudaErrorTrace();
	
	unsigned int *random_uints[2];
	float *random_floats[2];
	
	cudaMalloc((void**) &random_uints[0], sizeof(int) * 2 * POPULATION_SIZE);
	cudaMalloc((void**) &random_uints[1], sizeof(int) * POPULATION_SIZE);
	cudaMalloc((void**) &random_floats[0], sizeof(int) * POPULATION_SIZE * GENOME_LENGTH);
	cudaMalloc((void**) &random_floats[1], sizeof(int) * POPULATION_SIZE * GENOME_LENGTH);

	for(int i = 0; i < NUM_GENERATIONS; i++) {
		int local_max = -1, local_best;

		for(int k = 0; k < NUM_FOODS; k++) {
			h_foods[2 * k] = rand() % M;
			h_foods[2 * k + 1] = rand() % N;
		}

		/* Copy food positions from host to device */
		cudaMemcpy(d_foods, h_foods, food_size, cudaMemcpyHostToDevice);
		cudaErrorTrace();
		
		/* Copy host organism to device */
		cudaMemcpy(d_organism, organism, data_size, cudaMemcpyHostToDevice);
		
		/**
		Logic:
		Each organism is allocated a block,
		the threads in the blocks perform
		the operations for the organism 
		*/
		blocks = POPULATION_SIZE;
		threads = 128;

		evaluate<<<blocks, threads, sizeof(float) * GENOME_LENGTH>>>(d_organism, d_foods, d_fitness_score, GENOME_LENGTH);

		cudaMemcpy(fitness_score, d_fitness_score, sizeof(int) * POPULATION_SIZE, cudaMemcpyDeviceToHost);
		cudaErrorTrace();
		
		for(int j = 0; j < POPULATION_SIZE; j++) {
			if(local_max < fitness_score[j]) {
				local_max = fitness_score[j];
				local_best = j;
			}
		}

		// printing the genome of the best organism in the generation to the file.
		for(int k = 0; k < GENOME_LENGTH; k++) {
			fprintf(fout, "%f ", organism[local_best * GENOME_LENGTH + k]);
		}
		fprintf(fout, "\n");

		max_score = max(max_score, local_max);

		printf("Score after generation %d => local: %d | max: %d\n", i, local_max, max_score);

		int ga_blocks = POPULATION_SIZE;
		int ga_threads = GENOME_LENGTH;

		int selected = selection(0.15);

		cudaMemcpy(d_organism, organism, data_size, cudaMemcpyHostToDevice);		
		
		curandGenerate(prng, random_uints[0], 2 * POPULATION_SIZE);
		curandGenerate(prng, random_uints[1], POPULATION_SIZE);		
		
		crossover<<<POPULATION_SIZE - selected, GENOME_LENGTH>>>(random_uints[0], random_uints[1], d_organism, selected);

		curandGenerateUniform(prng, random_floats[0], POPULATION_SIZE * GENOME_LENGTH);
		curandGenerateNormal(prng, random_floats[1], POPULATION_SIZE * GENOME_LENGTH, 0.0, 1.0);
		
		mutate<<<ga_blocks, ga_threads>>>(random_floats[0], random_floats[1], d_organism, 1e-2);

		cudaMemcpy(organism, d_organism, data_size, cudaMemcpyDeviceToHost);
	}

	cudaFree(d_organism);
	cudaFree(d_fitness_score);
	cudaFree(d_foods);
	cudaFree(random_uints[0]);
	cudaFree(random_uints[1]);
	cudaFree(random_floats[0]);
	cudaFree(random_floats[1]);

	free(organism);

	return 0;
}