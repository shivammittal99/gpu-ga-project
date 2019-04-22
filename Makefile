all: snake-serial snake-parallel snake-super snake-test

snake-serial: snake-serial.cpp
	g++ -std=c++14 -o snake-serial snake-serial.cpp -lgraph

snake-parallel: snake-parallel.cu
	nvcc -std=c++14 -o snake-parallel snake-parallel.cu

snake-super: snake-super.cu
	nvcc -std=c++14 -G -g -o snake-super snake-super.cu -lcurand

snake-test: snake-test.cpp
	g++ -std=c++14 -o snake-test snake-test.cpp -lgraph

clean:
	@touch snake-serial snake-parallel snake-super snake-test
	@rm snake-serial snake-parallel snake-super snake-test