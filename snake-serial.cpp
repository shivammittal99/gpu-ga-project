/*	CS6023 GPU Programming
 	Project - Genetic Algorithm to optimise snakes game
 		Done By, 
 		Shivam Mittal, cs16b038
        Rachit Tibrewal, cs16b022
        R Sai Harshini, cs16b112
 	Serial Code
*/
#pragma once
#include <graphics.h>
#include <bits/stdc++.h>

using namespace std;
typedef pair<int, int> ii;

int POPULATION_SIZE = 4096;
int NUM_GENERATIONS = 200;

// Defining the parameters in each layer of the neural network.
int n = 24, m1 = 16, o = 4;

int GENOME_LENGTH;	//Genome length of each organism
float *organism;
std::random_device oracle{};
auto tempo = oracle();    
mt19937 rd{tempo};



// Neural Network Computation

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




// Genetic Algorithm
//1. Fitness Score computation

int *fitness_score = NULL, max_score;

// Utility funtion for evaluating the fitness function
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

//Function to evalate the fitness function.
int* evaluate(float *genes, int num_organisms, int generation_id, bool visualize, int foods[][2], int num_foods) {

	int *scores = (int *) malloc(sizeof(int) * num_organisms);

	for(int ind = 0; ind < num_organisms; ind++) {

		// for visalisation part.
		const int off_x = 10;
		const int off_y = 10;
		const int ss_x = 400;
		const int ss_y = 400;

		int ps_x = ss_x / M;
		int ps_y = ss_y / N;

		bool snakeIsAlive = true;
		bool foodEaten = true;
		ii food_pos;
		queue<pair<int,int>> snake; // Stores the coordinates along the length of the snake.
		int snake_init_length = 5;
		int init_x = rand()%(M-snake_init_length-2);
		int init_y = rand()%(M-snake_init_length-2);
		init_x = M/2;
		init_y = N/2;

		// Initial coordinates along the length of the snake.
		for(int i = 0; i<snake_init_length; i++) {
			snake.push(ii(i+init_x,0+init_y));
		}


		int maxiters = 3 * (M + N);
		int additers = 1*(M+N);
		int loops = maxiters;

		// directions.
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
			ii pos[8];
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
				//snake is not alive
				break;
			}
			if(visualize) {
				// Generate new graphics
				cleardevice(); 

				/// display the boundary
				rectangle(0+off_x,0+off_y,ss_x+off_x,ss_y+off_y);
				// display the food
				circle(off_x+food_pos.first*ps_x+ps_x/2, off_y+food_pos.second*ps_y+ps_y/2, ps_x/2);
				floodfill(off_x+food_pos.first*ps_x+ps_x/2, off_y+food_pos.second*ps_y+ps_y/2, RED);
				// display the snake
				snake_size = snake.size();
				for(int k=0; k < snake_size; k++) {
					ii x = snake.front();
					snake.pop();
					int st_x = off_x+x.first*ps_x;
					int st_y = off_y+x.second*ps_y;
					int en_x = st_x + ps_x;
					int en_y = st_y + ps_y;
					rectangle(st_x, st_y, en_x, en_y);
					snake.push(x);
				}

				outtextxy(ss_x * 1.1, ss_y / 2 - 20, &string("Generation: " + to_string(generation_id))[0]);
				outtextxy(ss_x * 1.1, ss_y / 2, &string("Score: " + to_string(score))[0]);
				outtextxy(ss_x * 1.1, ss_y / 2 + 20, &string("Max score: " + to_string(max_score))[0]);
				delay(10);
			}

		}while(snakeIsAlive && loops-- && fi < num_foods);

		if(visualize) {
			outtextxy(ss_x / 2 - 40, ss_y / 2, &string("Game Over")[0]);
		}

		scores[ind] = score;
	}

	return scores;
}


// Function to generate the initil genome for the all organisms.
// Each gene in the genome of the organsim corresponds weight of one of the edge in the neural network.
void createGenomes() {
	// length of genome in an organism is equla to the total no.of weights in the neural network.
	GENOME_LENGTH = n * m1 + m1 + m1 * o + o;
	
	organism = (float *) malloc(sizeof(float) * POPULATION_SIZE * GENOME_LENGTH);


	uniform_real_distribution<float> frand(-1, 1);

	for(int i = 0; i < POPULATION_SIZE; i++) {
		for(int j = 0; j < GENOME_LENGTH; j++) {
			organism[i * GENOME_LENGTH + j] = frand(rd);
		}
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
void crossover(int num_parents) {
	int total = num_parents;

	int parent[2];

	uniform_int_distribution<int> irand1(0, num_parents-1);
	uniform_int_distribution<int> irand2(0, GENOME_LENGTH-1);

	int pos;

	while(total < POPULATION_SIZE) {
		parent[0] = irand1(rd);
		parent[1] = irand1(rd);

		pos = irand2(rd);

		for(int i = 0; i <= pos; i++) {
			organism[total * GENOME_LENGTH + i] = organism[parent[0] * GENOME_LENGTH + i];
		}

		for(int i = pos+1; i < GENOME_LENGTH; i++) {
			organism[total * GENOME_LENGTH + i] = organism[parent[1] * GENOME_LENGTH + i];
		}

		total++;
	}
}

// Function to mutate the genomes of each organism. Mutation is one of the fundamental concept of genetic algorithms. 
void mutate(float mutation_rate) {
	uniform_real_distribution<float> frand1(0, 1);
	normal_distribution<float> frand2(0.0, 1.0);
	
	for(int i = 0; i < POPULATION_SIZE; i++) {
		for(int j = 0; j < GENOME_LENGTH; j++) {
			if(frand1(rd) < mutation_rate) // Making the proability of mutation very small.
			{
				organism[i * GENOME_LENGTH + j] += frand2(rd) / 5.0;
				organism[i + GENOME_LENGTH + j] = max(-1.0f, organism[i + GENOME_LENGTH + j]);
				organism[i + GENOME_LENGTH + j] = min(1.0f, organism[i + GENOME_LENGTH + j]);
			}
		}
	}
}


int main() 
{
	cout << tempo << endl;
	srand(42);

	// Graph for visualisation part.
	int gd = DETECT, gm; 
    initgraph(&gd, &gm, NULL);

	createGenomes();
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

		// Generating the food particles randomly, for the generation
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

		// printing the genome of the best organism in the generation to the file.
		for(int k = 0; k < GENOME_LENGTH; k++) {
			fprintf(fout, "%f ", organism[local_best * GENOME_LENGTH + k]);
		}
		fprintf(fout, "\n");

		free(evaluate(organism + local_best * GENOME_LENGTH, 1, i, false, foods, num_foods));

		max_score = max(max_score, local_max);

		printf("Score after generation %d => local: %d | max: %d\n", i, local_max, max_score);
		
		int selected = selection(0.15);
		crossover(selected);
		mutate(1e-2);
	}

	free(organism);

	closegraph();
	return 0;
}