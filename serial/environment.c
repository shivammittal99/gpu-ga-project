#include <stdlib.h>
#include "neuralnet.c"

int evaluate(float gene[]) {
	float inputs[2];

	int score = 0;

	do {
		inputs[0] = rand();
		inputs[1] = rand();
		forward(inputs, gene);
		score++;
	} while(((float)rand()) / ((float)RAND_MAX) > 0.5);

	return score;
}