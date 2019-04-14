all: snake-serial snake-parallel

snake-serial: snake-serial.cpp
	g++ -std=c++14 -o snake-serial snake-serial.cpp -lgraph

snake-parallel: snake-parallel.cu
	nvcc -std=c++14 -o snake-parallel snake-parallel.cu

clean:
	@touch snake-serial snake-parallel
	@rm snake-serial snake-parallel