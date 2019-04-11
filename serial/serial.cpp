#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "environment.hpp"
// #include "genome.h"
// #include "neuralnet.h"
const int POPULATION_SIZE = 128;

genome *organism;
int *score, max_score;

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
		if(frand() * score[i] / max_score > selection_cutoff) {
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
	srand(time(NULL));

	printf("Genome length: %d\n", GENOME_LENGTH);

	createGenomes();

	printf("Generation size: %d\n", POPULATION_SIZE);

	const int NUM_GENERATIONS = 20;

	score = (int *) malloc(sizeof(int) * POPULATION_SIZE);

	max_score = 0;

	FILE *fout = fopen("genomes.txt", "w");

	fprintf(fout, "NUM_GENERATIONS = %d\n", NUM_GENERATIONS);
	fprintf(fout, "POPULATION_SIZE = %d\n", POPULATION_SIZE);
	fprintf(fout, "GENOME_LENGTH = %d\n", GENOME_LENGTH);

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
	}

	printf("max score: %d\n", max_score);

	free(organism);
	return 0;
}