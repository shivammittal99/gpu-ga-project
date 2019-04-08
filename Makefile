flappy-serial: serial/serial.c serial/genome.c serial/environment.c serial/neuralnet.c
	gcc -o flappy-serial serial/serial.c -lm

clean: flappy-serial
	@rm flappy-serial