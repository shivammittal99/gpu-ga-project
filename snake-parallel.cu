#pragma once
#include <bits/stdc++.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

using namespace std;

int POPULATION_SIZE = 4096;
int NUM_GENERATIONS = 200;

int n = 24, m1 = 16, o = 4;

int GENOME_LENGTH;

float *organism;

/* Compute output of one fully connected layer
 * input: n
 * W: n * m
 * b: m
 * output: m
 */
float * dense(float input[], float W[], float b[], int n, int m) {
	float * output = (float *) malloc(m * sizeof(float));

	for(int i = 0; i < m; i++) {
		output[i] = 0;
		for(int j = 0; j < n; j++) {
			output[i] += W[j * m + i] * input[j];
		}
		output[i] += b[i];
	}

	return output;
}

/* Compute output of sigmoid layer
 * input: n
 * output: n
 */
float * sigmoid(float input[], int n) {
	float * output = (float *) malloc(n * sizeof(float));

	for(int i = 0; i < n; i++) {
		output[i] = 1.0 / (1.0 + exp(-input[i]));
	}

	return output;
}

/* Architecture of neural network
 * input: n => distance of closest object in directions
 * hidden layer: m1
 * output: o => direction to move in: straight, left or right
 */
int forward(float *input, float gene[]) {
	float *W1 = &gene[0];
	float *b1 = &gene[n * m1];
	float *W2 = &gene[n * m1 + m1];
	float *b2 = &gene[n * m1 + m1 + m1 * o];

	float *dense1 = dense((float *) input, W1, b1, n, m1);
	float *sigm1 = sigmoid(dense1, m1);
	free(dense1);
	float *dense2 = dense(sigm1, W2, b2, m1, o);
	free(sigm1);
	float *sigm2 = sigmoid(dense2, o);
	free(dense2);

	float maxm = sigm2[0];
	int res = 0;

	for(int i = 1; i < o; i++) {
		if(sigm2[i] > maxm) {
			maxm = sigm2[i];
			res = i;
		}
	}
	
	free(sigm2);

	return res;
}

int *fitness_score = NULL, max_score;

typedef pair<int, int> ii;

bool check(int u, int v, int i, int j) {
	if(u == 0 && v !=0) {
		if(i==u && j == v / abs(v)) return true;
	}
	else if(u!=0 && v == 0) {
		if(i == u/abs(u) && j == v) return true;
	}
	else if(u!=0 && v!=0) {
		if(i==u/abs(u) && j==v/abs(v)) return true;
	}
	return false;
}

const int M = 80;
const int N = 80;

int* evaluate(float *genes, int num_organisms, int generation_id, bool visualize, int foods[][2], int num_foods) {
	int *scores = (int *) malloc(sizeof(int) * num_organisms);
	for(int ind = 0; ind < num_organisms; ind++) {
		const int off_x = 10;
		const int off_y = 10;
		const int ss_x = 400;
		const int ss_y = 400;

		int ps_x = ss_x / M;
		int ps_y = ss_y / N;

		bool snakeIsAlive = true;
		bool foodEaten = true;
		ii food_pos;
		queue<pair<int,int>> snake;
		int snake_init_length = 5;
		int init_x = rand()%(M-snake_init_length-2);
		int init_y = rand()%(M-snake_init_length-2);
		init_x = M/2;
		init_y = N/2;
		for(int i = 0; i<snake_init_length; i++) {
			snake.push(ii(i+init_x,0+init_y));
		}
		int maxiters = 3 * (M + N);
		int additers = 1*(M+N);
		int loops = maxiters;
		// 1 north:0,-1
		// 2 south:0,1
		// 3 east: 1,0
		// 4 west:-1,0
		int snake_motion = 3; 
		ii dir = ii(1,0);
		int score = 0;
		int input_dim = n;
		int t = input_dim / 2;
		int fi = 0;
		do
		{
			fflush(stdout);
			if(foodEaten) {
				food_pos =  ii(foods[fi][0],foods[fi][1]);
				fi++;
				foodEaten = false;
			}
			ii head = snake.back();
			int x = head.first;
			int y = head.second;
			
			int snake_size = snake.size();
			
			//ii pos[8];
			float dist[8][3];
			for(int i=0;i < 8;i++) {
				for(int j = 0; j < 3; j++) {
					dist[i][j] = 2*max(M, N);
				}
			}
			int k = 0;
			for(int i=-1;i<=1;i++) {
				for(int j=-1; j<=1;j++) {
					if (i == 0 && j == 0 ) {
						continue;
					} else if(i == 0) {
						dist[k][0] = (j > 0) * N - j * y;
					} else if(j == 0) {
						dist[k][0] = (i > 0) * M - i * x;
					} else {
						dist[k][0] = min((i > 0) * M - i * x, (j > 0) * N - j * y);
					}

					int u,v;
					// food_pos
					u = food_pos.first - x;
					v = food_pos.second - y;
					if(check(u,v,i,j)) {
						if(abs(dist[k][1]) > float(abs(u)+abs(v)) / (abs(i) + abs(j))) {
							dist[k][1] = float(abs(u)+abs(v)) / (abs(i) + abs(j));
						}
					}
					for(int ti=0; ti < snake_size; ti++) {
						ii haha = snake.front();
						snake.pop();
						u = haha.first - x;
						v = haha.second - y;
						if(check(u,v,i,j)) {
							if(abs(dist[k][2]) > float(abs(u)+abs(v))/(abs(i)+abs(j))) {
								dist[k][2] = float(abs(u)+abs(v))/(abs(i)+abs(j));
							}
						}
						
						snake.push(haha);
					}
					k++;						
				}
			}
			
			/**
			 * 0 straight
			 * 1 left
			 * 2 right
			 */
			// 1 north:0,-1
			// 2 south:0,1
			// 3 east: 1,0
			// 4 west:-1,0

			// for(int i = 0; i < 8; i++) {
			// 	printf("i: %d | dist: %f\n", i, dist[i]);
			// }

			int com = forward(&dist[0][0], genes + ind * GENOME_LENGTH);
			
			if(com == 0) {
				// no change to direction
			}
			else if(com == 1) {
				if(snake_motion == 1) {
					// change to west
					snake_motion = 4;
					dir = ii(-1,0);
				}
				else if(snake_motion == 2) {
					// change to east
					snake_motion = 3;
					dir = ii(1,0);
				}
				else if(snake_motion == 3) {
					// change to north
					snake_motion = 1;
					dir = ii(0,-1);
				}
				else if(snake_motion == 4) {
					// change to south
					snake_motion = 2;
					dir = ii(0,1);
				}
			}
			else if(com == 2){
				if(snake_motion == 1) {
					// change to east
					snake_motion = 3;
					dir = ii(1,0);
				}
				else if(snake_motion == 2) {
					// change to west
					snake_motion = 4;
					dir = ii(-1,0);
				}
				else if(snake_motion == 3) {
					// change to south
					snake_motion = 2;
					dir = ii(0,1);
				}
				else if(snake_motion == 4) {
					// change to north
					snake_motion = 1;
					dir = ii(0,-1);
				}
			}
			else if(com == 3) {
				snakeIsAlive = false;
				break;
			}
			
			// check if the snake eats the food in the next move
			head = ii(head.first+dir.first, head.second+dir.second); 
			snake.push(head);

			// move the snake in the direction
			if(head != food_pos) {
				snake.pop();

			}
			else {
				score += 1;
				loops += additers;
				foodEaten = true;
			}


			// check if the snake crosses any boundaries
			x = head.first;
			y = head.second;
			if(x<0||y<0||x>=M||y>=N) {
				// crossed the boundart game over
				snakeIsAlive = false;
				break;
			}
			// check if the snake eats it self
			snake_size = snake.size();
			for(int i=0; i < snake_size; i++) {
				ii haha = snake.front();
				snake.pop();
				if(i < snake_size-1 && haha.first == x && haha.second == y) {
					snakeIsAlive = false;
					break;
				}            
				snake.push(haha);
			}
			if(!snakeIsAlive) {
				break;
			}

		}while(snakeIsAlive && loops-- && fi < num_foods);


		scores[ind] = score;
	}

	return scores;
}



__global__ void createGenomes(float* organism)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	thrust::default_random_engine randEng;
	thrust::uniform_real_distribution<float> frand(-1, 1);
	randEng.discard(i);

	// if(threadIdx.x < genomeLength)	// within the genome length of the organism allocated to the block
	organism[i] = frand(randEng);
}

__global__ void selection_parallel(float* current_generation, float* new_generation, int *index_list, int selected, int genomeLength )
{
	if(blockIdx.x<selected && threadIdx.x<genomeLength)
	{
		int i = index_list[blockIdx.x] * genomeLength + threadIdx.x;
		int j = blockIdx.x *genomeLength + threadIdx.x;

		new_generation[j] = current_generation[i];
	} 
}

int selection(float selection_cutoff) {
	int selected = int(selection_cutoff*POPULATION_SIZE);
	
	size_t size1 = sizeof(float) * POPULATION_SIZE * GENOME_LENGTH;
	size_t size2 = sizeof(int) * selected ;
	float *new_generation = (float *) malloc(size1);
	int *index_list = (int *) malloc(size2);

	float* d_new_generation;
	float* d_current_generation;
	int* d_index_list;
  	cudaMalloc(&d_new_generation,size1);	
  	cudaMalloc(&d_current_generation,size1);
  	cudaMalloc(&d_index_list,size2);

	ii list[POPULATION_SIZE];
	
	for(int i = 0; i < POPULATION_SIZE; i++) {
		list[i] = {-fitness_score[i], i};
	}

	sort(list, list + POPULATION_SIZE);

	for(int i = 0 ; i<selected; i++)
	{
		index_list[i] = list[i].second;
	}

	cudaMemcpy(d_index_list, index_list, size2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_current_generation, organism, size1,cudaMemcpyHostToDevice);

	int threadsPerBlock = 512;
	int blocksPerGrid = 1024;

	selection_parallel<<<blocksPerGrid, threadsPerBlock>>>(d_current_generation, d_new_generation, d_index_list, selected, GENOME_LENGTH);
	cudaDeviceSynchronize();
	cudaMemcpy(new_generation, d_new_generation, size1,cudaMemcpyDeviceToHost);

	free(organism);
	organism = new_generation;
	return selected;
}


//each thread handles one organism
__global__ void crossover_parallel(float* current_generation, int num_parents, int genomeLength, int population_size )
{
	int idx = (blockDim.x * blockIdx.x + threadIdx.x)+num_parents;
	if(idx < population_size)
	{
		thrust::default_random_engine randEng;
		thrust::uniform_int_distribution<int> irand1(0, num_parents-1);
		thrust::uniform_int_distribution<int> irand2(0, genomeLength-1);
		randEng.discard(idx);

		int parent_0 = irand1(randEng);
		int parent_1 = irand1(randEng);

		int pos = irand2(randEng);

		for(int i = 0; i <= pos; i++) 
		{
			current_generation[idx * genomeLength + i] = current_generation[parent_0 * genomeLength + i];
		}

		for(int i = pos+1; i < genomeLength; i++) 
		{
			current_generation[idx * genomeLength + i] = current_generation[parent_1 * genomeLength + i];
		}
	} 
}

void crossover(int num_parents) 
{

	size_t size1 = sizeof(float) * POPULATION_SIZE * GENOME_LENGTH;

	float* d_current_generation;	
  	cudaMalloc(&d_current_generation,size1);

  	cudaMemcpy(d_current_generation, organism, size1,cudaMemcpyHostToDevice);

  	int threadsPerBlock = 256;
	int blocksPerGrid = 16;

	crossover_parallel<<<blocksPerGrid, threadsPerBlock>>>(d_current_generation, num_parents, GENOME_LENGTH, POPULATION_SIZE);
	cudaDeviceSynchronize();
	cudaMemcpy(organism, d_current_generation, size1,cudaMemcpyDeviceToHost);
}

__global__ void mutate_parallel(float* organism, int genomeLength ,int mutation_rate)
{
	int i = blockIdx.x * genomeLength + threadIdx.x;

	thrust::default_random_engine randEng;
	thrust::uniform_real_distribution<float> frand1(0, 1);
	thrust::normal_distribution<float> frand2(0.0, 1.0);
	randEng.discard(i);

	if(threadIdx.x < genomeLength)	// within the genome length of the organism allocated to the block
	{
		if(frand1(randEng) < mutation_rate) 
		{
			organism[i] += frand2(randEng) / 5.0;
			organism[i] = max(-1.0f, organism[i]);
			organism[i] = min(1.0f, organism[i]);
		}
	}
}

void mutate(float mutation_rate) {

	size_t size1 = sizeof(float) * POPULATION_SIZE * GENOME_LENGTH;

	float* d_organism;
  	cudaMalloc(&d_organism,size1);

  	cudaMemcpy(d_organism, organism, size1,cudaMemcpyHostToDevice);

  	int threadsPerBlock = 512;
	int blocksPerGrid = 4096;

	mutate_parallel<<<blocksPerGrid, threadsPerBlock>>>(d_organism,GENOME_LENGTH,mutation_rate);
	cudaDeviceSynchronize();
	cudaMemcpy(organism, d_organism, size1,cudaMemcpyDeviceToHost);
}

int main() {
	srand(time(NULL));

	
	GENOME_LENGTH = n * m1 + m1 + m1 * o + o;
	size_t size1 = sizeof(float) * POPULATION_SIZE * GENOME_LENGTH;
	organism = (float *) malloc(size1);

	float* d_organism;
  	cudaMalloc(&d_organism,size1);	

	int threadsPerBlock = 512;
	int blocksPerGrid = 4096;

	createGenomes<<<blocksPerGrid, GENOME_LENGTH>>>(d_organism);
	cudaDeviceSynchronize();
	cudaMemcpy(organism, d_organism, size1,cudaMemcpyDeviceToHost);
	double mu = 0;
	double sigma = 0;
	int L = POPULATION_SIZE*GENOME_LENGTH;
	cout  << "Finding the mean and sigma" << endl;
	for(int i=0;i < L; i++) {
		mu += *(organism+i);
	}
	mu = mu / L;
	for(int i=0; i < L; i++) {
		sigma += pow((*(organism+i)-mu),2);
	}
	sigma /= L;
	printf("mean: %lf | sigma: %lf\n", mu, sigma);
	printf("Genome length: %d\n", GENOME_LENGTH);
	printf("Generation size: %d\n", POPULATION_SIZE);

	max_score = 0;

	FILE *fout = fopen("genomes.txt", "w");

	fprintf(fout, "NUM_GENERATIONS = %d\n", NUM_GENERATIONS);
	fprintf(fout, "POPULATION_SIZE = %d\n", POPULATION_SIZE);
	fprintf(fout, "GENOME_LENGTH = %d\n", GENOME_LENGTH);
		
	for(int i = 0; i < NUM_GENERATIONS; i++) {
		int local_max = -1, local_best;

		if(fitness_score != NULL) {
			free(fitness_score);
		}
		int num_foods = 100;
		int foods[num_foods][2];
		for(int k=0; k < num_foods; k++) {
			foods[k][0] = (rand())%M;
			foods[k][1] = (rand())%N;
		}
		fitness_score = evaluate(organism, POPULATION_SIZE, i, false, foods, num_foods);

		for(int j = 0; j < POPULATION_SIZE; j++) {
			if(local_max < fitness_score[j]) {
				local_max = fitness_score[j];
				local_best = j;
			}
		}

		for(int k = 0; k < GENOME_LENGTH; k++) {
			fprintf(fout, "%f ", organism[local_best * GENOME_LENGTH + k]);
		}
		fprintf(fout, "\n");

		free(evaluate(organism + local_best * GENOME_LENGTH, 1, i, true, foods, num_foods));

		max_score = max(max_score, local_max);

		printf("Score after generation %d => local: %d | max: %d\n", i, local_max, max_score);
		
		int selected = selection(0.15);
		crossover(selected);
		mutate(1e-2);
	}

  	cudaFree(d_organism);
	free(organism);
	return 0;
}