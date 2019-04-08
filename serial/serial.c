#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "environment.c"
#include "genome.c"

genome *organism;
float *score;

void createGenomes(int population_size) {
	organism = (genome *) malloc(sizeof(genome) * population_size);

	for(int i = 0; i < population_size; i++) {
		for(int j = 0; j < GENOME_LENGTH; j++) {
			organism[i].genes[j] = ((float)rand()) / ((float)RAND_MAX) * 0.2 - 0.1;
		}
	}
}

void selection() {

}

void crossover() {

}

void mutate() {
	
}

int main() {
	srand(time(NULL));

	printf("Genome length: %d\n", GENOME_LENGTH);

	const int POPULATION_SIZE = 128;

	createGenomes(POPULATION_SIZE);

	printf("Generation size: %d\n", POPULATION_SIZE);

	const int NUM_GENERATIONS = 20;

	score = (float *) malloc(sizeof(float) * POPULATION_SIZE);

	float max_score = 0;

	for(int i = 0; i < NUM_GENERATIONS; i++) {
		for(int j = 0; j < POPULATION_SIZE; j++) {
			score[j] = evaluate(organism[i].genes);
			max_score = fmax(max_score, score[j]);
			selection();
			crossover();
			mutate();
		}
	}

	printf("max score: %f\n", max_score);

	return 0;
}