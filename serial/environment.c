#include <stdlib.h>
#include "neuralnet.c"

float evaluate(float gene[]) {
	float inputs[2] = {rand(), rand()};

	return forward(inputs, gene);
}