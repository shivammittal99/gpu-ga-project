#include <math.h>

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