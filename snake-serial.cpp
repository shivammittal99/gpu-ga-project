#include <bits/stdc++.h>

using namespace std;

int POPULATION_SIZE = 32;
int NUM_GENERATIONS = 100;

int n = 7, m = 64, o = 3;

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
		output[i] = 1.0 / (1.0 + exp(input[i]));
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

int* evaluate(float *genes, int num_organisms, bool visualize) {
	int *scores = (int *) malloc(sizeof(int) * num_organisms);

	for(int i = 0; i < num_organisms; i++) {
		scores[i] = rand()%10000;
	}

	return scores;
}

int *score = NULL, max_score;

void createGenomes(int field_of_view, int hidden_layer_size, int num_outputs) {
	GENOME_LENGTH = field_of_view * hidden_layer_size + hidden_layer_size + hidden_layer_size * num_outputs + num_outputs;
	
	organism = (float *) malloc(sizeof(float) * POPULATION_SIZE * GENOME_LENGTH);

	random_device rd;
	uniform_real_distribution<float> frand(-0.1, 0.1);

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
		if(frand(rd) * score[i] / (max_score + 1) > selection_cutoff) {
			copy(organism + i * GENOME_LENGTH, organism + (i + 1) * GENOME_LENGTH, new_generation + selected * GENOME_LENGTH);
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
			organism[total * GENOME_LENGTH + i] = organism[parent[rand() % 2] * GENOME_LENGTH + i];
		}

		total++;
	}
}

void mutate(float mutation_rate) {
	random_device rd;
	uniform_real_distribution<float> frand1(0, 1);
	uniform_real_distribution<float> frand2(-0.1, 0.1);
	
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

	createGenomes(n * n * 4, m, o);

	printf("Genome length: %d\n", GENOME_LENGTH);
	printf("Generation size: %d\n", POPULATION_SIZE);

	max_score = 0;

	FILE *fout = fopen("genomes.txt", "w");

	fprintf(fout, "NUM_GENERATIONS = %d\n", NUM_GENERATIONS);
	fprintf(fout, "POPULATION_SIZE = %d\n", POPULATION_SIZE);
	fprintf(fout, "GENOME_LENGTH = %d\n", GENOME_LENGTH);

	for(int i = 0; i < NUM_GENERATIONS; i++) {
		int local_max = 0, local_best;

		if(score != NULL) {
			free(score);
		}

		score = evaluate(organism, POPULATION_SIZE, false);

		for(int j = 0; j < POPULATION_SIZE; j++) {
			if(local_max < score[j]) {
				local_max = score[j];
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
	return 0;
}