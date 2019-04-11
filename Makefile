flappy-serial: serial/serial.cpp serial/genome.h serial/environment.hpp serial/neuralnet.h
	g++ -o flappy-serial serial/serial.cpp -lm

clean: flappy-serial
	@rm flappy-serial