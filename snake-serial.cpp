#include <graphics.h>
#include <bits/stdc++.h>

using namespace std;

int POPULATION_SIZE = 64;
int NUM_GENERATIONS = 100;

int n = 9, m = 64, o = 3;

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
 * input: n * n => field of view of snake
 * hidden layer: m
 * output: o => direction to move in: straight, left or right
 */
int forward(int *input, float gene[], int n, int m, int o) {
	float *one_hot = (float *) malloc(sizeof(float) * n * n * 4);

	memset(one_hot, 0, sizeof(one_hot));

	for(int i = 0; i < n * n; i++) {
		one_hot[i * 4 + input[i]] = 1;
	}

	float *W1 = &gene[0];
	float *b1 = &gene[n * n * 4 * m];
	float *W2 = &gene[n * n * 4 * m + m];
	float *b2 = &gene[n * n * 4 * m + m + m * o];

	float *dense1 = dense(one_hot, W1, b1, n * n * 4, m);
	float *sigm1 = sigmoid(dense1, m);
	free(dense1);
	float *dense2 = dense(sigm1, W2, b2, m, o);
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

typedef pair<int, int> ii;

int* evaluate(float *genes, int num_organisms, bool visualize) {
	int *scores = (int *) malloc(sizeof(int) * num_organisms);

	for(int ind = 0; ind < num_organisms; ind++) {
		const int off_x = 10;
		const int off_y = 10;
		const int ss_x = 400;
		const int ss_y = 400;
		const int M = 20;
		const int N = 20;
		int ps_x = ss_x / M;
		int ps_y = ss_y / N;

		bool snakeIsAlive = true;
		bool foodEaten = true;
		ii food_pos;
		queue<pair<int,int>> snake;
		int snake_init_length = 5;
		for(int i = 0; i<snake_init_length; i++) {
			snake.push(ii(i,0));
		}
		int maxiters = 50;
		// 1 north:0,-1
		// 2 south:0,1
		// 3 east: 1,0
		// 4 west:-1,0
		int snake_motion = 3; 
		ii dir = ii(1,0);
		int score = 0;
		int input_dim = n;
		int t = input_dim / 2;
		do
		{
			// printf("%d\n",(int)snake.size());
			// maxiters = 50;
			fflush(stdout);
			if(foodEaten) {
				food_pos =  ii(rand()%M, rand()%N);
				foodEaten = false;
			}
			ii head = snake.back();
			int x = head.first;
			int y = head.second;
			// cout << "0" << endl;
			// create the input for Neural Network        
			int in[input_dim][input_dim];
			for(int i = x-t, i1 = 0; i <= x+t; i++, i1++) {
				for(int j=y-t, j1=0; j <= y+t; j++, j1++) {
					in[i1][j1] = 0;
					if(i < 0 || j < 0 || i >= M || j >= N) {
						in[i1][j1] = 1;
					}
					else if(i == food_pos.first && j == food_pos.second) {
						in[i1][j1] = 3;
					}
				}
			}
			// cout << "1" << endl;
			int snake_size = snake.size();
			for(int i=0; i < snake_size; i++) {
				ii haha = snake.front();
				snake.pop();
				if(haha.first>=x-t && haha.first <= x+t && haha.second>=y-t && haha.second<=y+t) {
					in[haha.first-(x-t)][haha.second-(y-t)] = 2;
				}
				snake.push(haha);
			}
			// cout << "Created input for neural network" << endl;
			// Take the input here
			// get the motion direction
			/**
			 * Code here
			 */
			/**
			 * 0 straight
			 * 1 left
			 * 2 right
			 */
			// 1 north:0,-1
			// 2 south:0,1
			// 3 east: 1,0
			// 4 west:-1,0
			int com = forward(&in[0][0], genes + ind * GENOME_LENGTH, n, m, o);
			
			// cout << "ind = " << ind << " | com = " << com << endl;
			
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
			// cout << "Done with motion direction" << endl;
			// check if the snake eats the food in the next move
			head = ii(head.first+dir.first, head.second+dir.second); 

			// move the snake in the direction
			if(head != food_pos) {
				snake.pop();

			}
			else {
				score += maxiters;
				foodEaten = true;
			}
			snake.push(head);

			// check if the snake crosses any boundaries
			x = head.first;
			y = head.second;
			if(x<0||y<0||x>=M||y>=N) {
				// crossed the boundart game over
				// cout << x << " " << y << " " << M << " " << N << endl;
				printf("Game Over");
				snakeIsAlive = false;
				break;
			}
			// check if the snake eats it self
			snake_size = snake.size();
			for(int i=0; i < snake_size; i++) {
				ii haha = snake.front();
				snake.pop();
				if(haha != head && haha.first == x && haha.second == y) {
					// cout << i << " "<< x << " " << y << " " << haha.first << " " << haha.second << endl;
					snakeIsAlive = false;
					break;
				}            
				snake.push(haha);
			}
			if(!snakeIsAlive) {
				// cout << "snake is not alive" << endl;
				break;
			}
			if(visualize) {
				// Generate new graphics
				cleardevice(); 

				/// display the boundary
				rectangle(0+off_x,0+off_y,ss_x+off_x,ss_y+off_y);
				// display the food
				circle(off_x+food_pos.first*ps_x+ps_x/2, off_y+food_pos.second*ps_y+ps_y/2,ps_x/2);
				// display the snake
				snake_size = snake.size();
				for(int k=0; k < snake_size; k++) {
					ii x = snake.front();
					snake.pop();
					int st_x = off_x+x.first*ps_x;
					int st_y = off_y+x.second*ps_y;
					int en_x = st_x + ps_x;
					int en_y = st_y + ps_y;
					rectangle(st_x,st_y,en_x,en_y);
					snake.push(x);
				}
				delay(20);
			}

		}while(snakeIsAlive && maxiters--);

		scores[ind] = score;
	}

	return scores;
}

int *fitness_score = NULL, max_score;

void createGenomes(int field_of_view, int hidden_layer_size, int num_outputs) {
	GENOME_LENGTH = field_of_view * hidden_layer_size + hidden_layer_size + hidden_layer_size * num_outputs + num_outputs;
	
	organism = (float *) malloc(sizeof(float) * POPULATION_SIZE * GENOME_LENGTH);

	random_device rd;
	uniform_real_distribution<float> frand(-1, 1);

	for(int i = 0; i < POPULATION_SIZE; i++) {
		for(int j = 0; j < GENOME_LENGTH; j++) {
			organism[i * GENOME_LENGTH + j] = frand(rd);
		}
	}
}

int selection(float selection_cutoff) {
	int selected = 0;
	float *new_generation = (float *) malloc(sizeof(float) * POPULATION_SIZE * GENOME_LENGTH);
	random_device rd;
	uniform_real_distribution<float> frand(0, 1);

	for(int i = 0; i < POPULATION_SIZE; i++) {
		if(frand(rd) * fitness_score[i] / (max_score + 1) > selection_cutoff) {
			copy(organism + i * GENOME_LENGTH, organism + (i + 1) * GENOME_LENGTH, new_generation + selected * GENOME_LENGTH);
			selected++;
		}
	}

	while(selected < 8) {
		int temp = rand()%POPULATION_SIZE;
		copy(organism + temp * GENOME_LENGTH, organism + (temp + 1) * GENOME_LENGTH, new_generation + selected * GENOME_LENGTH);
		selected++;
	}

	free(organism);
	organism = new_generation;
	return selected;
}

void crossover(int num_parents) {
	int total = num_parents;

	int parent[2];

	while(total < POPULATION_SIZE) {
		parent[0] = rand() % num_parents;
		parent[1] = rand() % num_parents;

		for(int i = 0; i < GENOME_LENGTH; i++) {
			organism[total * GENOME_LENGTH + i] = organism[parent[rand() % 2] * GENOME_LENGTH + i];
		}

		total++;
	}
}

void mutate(float mutation_rate) {
	random_device rd;
	uniform_real_distribution<float> frand1(0, 1);
	uniform_real_distribution<float> frand2(-1, 1);
	
	for(int i = 0; i < POPULATION_SIZE; i++) {
		for(int j = 0; j < GENOME_LENGTH; j++) {
			if(frand1(rd) < mutation_rate) {
				organism[i * GENOME_LENGTH + j] = frand2(rd);
			}
		}
	}
}

int main() {
	srand(time(NULL));

	int gd = DETECT, gm; 
    initgraph(&gd, &gm, NULL);

	createGenomes(n * n * 4, m, o);

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

		fitness_score = evaluate(organism, POPULATION_SIZE, false);

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

		free(evaluate(organism + local_best * GENOME_LENGTH, 1, true));

		max_score = max(max_score, local_max);

		printf("Score after generation %d => local: %d | max: %d\n", i, local_max, max_score);
		int selected = selection(0.15);

		crossover(selected);
		mutate(1e-3);
	}

	free(organism);

	closegraph();
	return 0;
}