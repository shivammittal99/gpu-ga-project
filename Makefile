flappy-serial: flappyBird_serial.cpp
	g++ -o flappy-serial flappyBird_serial.cpp -lm

clean: flappy-serial
	@rm flappy-serial
