#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define GENOME_LENGTH 65

using namespace std;

//genome
typedef struct genome {
	float genes[GENOME_LENGTH];
} genome;

//neuralnet
float * dense(float input[], float W[], float b[], int n, int m) {
	float * output = (float *) malloc(m * sizeof(float));

	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			output[i] += W[j * m + i] * input[j];
		}
		output[i] += b[i];
	}

	return output;
}

float * sigmoid(float input[], int n) {
	float * output = (float *) malloc(n * sizeof(float));

	for(int i = 0; i < n; i++) {
		output[i] = 1.0 / (1.0 + exp(input[i]));
	}

	return output;
}

/* Architecture of neural network
 * input: 2 => bird's height, closest pipe's height
 * hidden layer: 10
 * output: 1 => probability of jumping
 */
float forward(float input[], float gene[]) {
	float *W1 = &gene[0];
	float *b1 = &gene[2 * 16];
	float *W2 = &gene[2 * 16 + 16];
	float *b2 = &gene[2 * 16 + 16 + 16 * 1];

	float *dense1 = dense(input, W1, b1, 2, 16);
	float *sigm1 = sigmoid(dense1, 16);
	free(dense1);
	float *dense2 = dense(sigm1, W2, b2, 16, 1);
	free(sigm1);
	float *sigm2 = sigmoid(dense2, 1);
	free(dense2);
	float prob = *sigm2;
	free(sigm2);

	return prob;
}

//environment
struct Pipe 
{
    float x = 0;
    float y = 0;
    float width = 50;
    float height = 40;
    float speed = 3;

    void update() 
    {
        this->x -= this->speed;
    }

    bool isOut() 
    {
        if(this->x + this->width < 0) {
            return true;
        }
    }
};

struct Bird {
    float x;
    float y;
    float width = 40;
    float height = 30;
    bool alive = true;
    float gravity = 0;
    float velocity = 0.3;
    float jump = -6;

    void flap() 
    {
        this->gravity = this->jump;
    }

    void update() 
    {
        this->gravity += this->velocity;
        this->y += this->gravity;
    }

    bool isDead(float height, const vector<Pipe>& pipes) 
    {
        if (this->y >= height || this->y + this->height <= 0) {
            return true;
        }

        for(auto pipe : pipes) {
            if(!(this->x > pipe.x + pipe.width || this->x+this->width < pipe.x || this->y > pipe.y + pipe.height || this->y + this->height < pipe.y)) {
                return true;
            }
        }
        return false;   
    }
};

class Game 
{
public:
    Game() 
    {
        cscore = 10000000;
    }

    Game(int cscore_) {
        cscore = cscore_;
    }

	int* start(genome* organisms_, int population_size) 
    {
		organisms = organisms_;
        scores = (int*)malloc(sizeof(int)*population_size);
		pipes.clear();
		birds.clear();
		score = 0;
		spawnInterval = 90;
		interval = 0;
		for(int i=0;i < population_size; i++) 
        {
			birds.push_back(Bird());
		}

		generation++;
		alives = birds.size();
		while(!isItEnd()) {
			update();
		}
		return scores;
	}

	bool isItEnd() 
    {
		for(int i = 0; i < birds.size(); i++) {
			if(birds[i].alive) {
				return false;
			}
		}
		return true;
	}

    void update() 
    {
        backgroundx += backgroundSpeed;
        float nextHoll = 0;
        if (birds.size() > 0) {
            for (int i = 0; i < pipes.size(); i+=2) {
                if (pipes[i].x + pipes[i].width > birds[0].x) {
                    nextHoll = pipes[i].height  / height;
                    break;
                }
            }
        }

        for(int i = 0; i < birds.size(); i++) {
            if(birds[i].alive) {
                float inputs[2];
                inputs[0] = birds[i].y / height;
                inputs[1] =  nextHoll;
                float res = forward(inputs,organisms[i].genes);
                if(res > 0.5) {
                    birds[i].flap();
                }
                birds[i].update();
                if(birds[i].isDead(height, pipes)) {
                    birds[i].alive = false;
                    alives--;
                }
				else {
					scores[i]++;
				}
            }
        }

        for(int i = 0; i < pipes.size(); i++) 
        {
            pipes[i].update();
            if(pipes[i].isOut()) {
                pipes.erase(pipes.begin()+i);
                i--;
            }
        }

        if(interval == 0) 
        {
            float deltaBord = 50;
            float pipeHoll = 120;
            float hollPosition = round(((rand()%1000)/1000.)*(height - deltaBord*2-pipeHoll)) + deltaBord;
            Pipe a;
            a.x = width;
            a.y = 0;
            a.height = hollPosition;
            Pipe b;
            b.x = width;
            b.y = hollPosition+pipeHoll;
            b.height = height;
            pipes.push_back(a);
            pipes.push_back(b);
            interval++;
            if(interval == spawnInterval) {
                interval = 0;
            }
        }
    }

private:
    vector<Pipe> pipes;
    vector<Bird> birds;
    int score = 0;
    int width = 400;
    int height = 800;
    int spawnInterval = 90;
    int interval = 0;
    int alives = 0;
    int generation = 0;
    float backgroundSpeed = 0.5;
    float backgroundx = 0;
	int* scores;
    int maxScore = 0;
    int cscore;
    genome* organisms;
};

// Genetic Algorithm
const int POPULATION_SIZE = 128;

genome *organism;
int *score, maximum_score;

float frand() {
	return ((float)rand()) / ((float)RAND_MAX);
}

void createGenomes() {
	organism = (genome *) malloc(sizeof(genome) * POPULATION_SIZE);

	for(int i = 0; i < POPULATION_SIZE; i++) {
		for(int j = 0; j < GENOME_LENGTH; j++) {
			organism[i].genes[j] = frand() * 0.2 - 0.1;
		}
	}
}

int selection(float selection_cutoff) {
	int selected = 0;
	genome *new_generation = (genome *) malloc(sizeof(genome) * POPULATION_SIZE);

	for(int i = 0; i < POPULATION_SIZE; i++) {
		if(frand() * score[i] / maximum_score > selection_cutoff) {
			new_generation[selected] = organism[i];
			selected++;
		}
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
			organism[total].genes[i] = organism[parent[rand() % 2]].genes[i];
		}

		total++;
	}
}

void mutate(float mutation_rate) {
	for(int i = 0; i < POPULATION_SIZE; i++) {
		for(int j = 0; j < GENOME_LENGTH; j++) {
			if(frand() < mutation_rate) {
				organism[i].genes[j] = frand() * 0.2 - 0.1;
			}
		}
	}
}

int main() {
	int ceiling_score = 100000;
	Game gg(ceiling_score);
	clock_t start, end;
	srand(time(NULL));

	printf("Genome length: %d\n", GENOME_LENGTH);

	createGenomes();

	printf("Generation size: %d\n", POPULATION_SIZE);

	const int NUM_GENERATIONS = 20;

	score = (int *) malloc(sizeof(int) * POPULATION_SIZE);

	maximum_score = 0;

	FILE *fout = fopen("genomes.txt", "w");

	fprintf(fout, "NUM_GENERATIONS = %d\n", NUM_GENERATIONS);
	fprintf(fout, "POPULATION_SIZE = %d\n", POPULATION_SIZE);
	fprintf(fout, "GENOME_LENGTH = %d\n", GENOME_LENGTH);


	start = clock();
	for(int i = 0; i < NUM_GENERATIONS; i++) {
		free(score);
		score = gg.start(organism, POPULATION_SIZE);
		// for(int j = 0; j < POPULATION_SIZE; j++) {
			// for(int k = 0; k < GENOME_LENGTH; k++) {
			// 	fprintf(fout, "%f ", organism[j].genes[k]);
			// }
			// fprintf(fout, "\n");
			// score[j] = evaluate(organism[j].genes);
			// max_score = max_score > score[j] ? max_score : score[j];
		// }
		int max_score = 0;
		int idx = -1;
		for(int j=0; j < POPULATION_SIZE; j++) {
			if(score[j] >= max_score) {
				idx = j;
				max_score = score[j];
			}
		}
		fprintf(fout,"Best of Gen %d :\n",i);
		for(int k = 0; k < GENOME_LENGTH; k++) {
			fprintf(fout, "%f ", organism[idx].genes[k]);
		}
		fprintf(fout, "\n");
		printf("Score after generation %d is %d\n", i, max_score);
		int selected = selection(0.15);
		crossover(selected);
		mutate(1e-3);
		maximum_score = max(maximum_score,max_score);
	}

	end = clock();
	float time_taken = (float)(end-start)/CLOCKS_PER_SEC;

	printf("max score: %d\n", maximum_score);
	printf("Time taken on CPU : %f\n",time_taken);

	free(organism);
	return 0;
}