all: snake-serial snake-test snake-parallel

snake-serial: snake-serial.cpp
	g++ -std=c++14 -o snake-serial snake-serial.cpp -lgraph

snake-test: snake-test.cpp
	g++ -std=c++14 -o snake-test snake-test.cpp -lgraph

snake-parallel: snake-parallel.cu
	nvcc -std=c++14 -o snake-parallel snake-parallel.cu -lcurand

clean:
	@touch snake-serial snake-parallel snake-test
	@rm snake-serial snake-parallel snake-test