// #pragma once
#include <bits/stdc++.h>
#include "genome.h"
#include "neuralnet.h"
using namespace std;
struct Pipe {
    float x = 0;
    float y = 0;
    float width = 50;
    float height = 40;
    float speed = 3;
    void update() {
        this->x -= this->speed;
    }
    bool isOut() {
        if(this->x + this->width < 0) {
            return true;
        }
    }
};
// class Genome {
//     float compute(const vector<float>& inputs) {
//         return 0.5;
//     }
// };
struct Bird {
    float x;
    float y;
    float width = 40;
    float height = 30;
    bool alive = true;
    float gravity = 0;
    float velocity = 0.3;
    float jump = -6;
    void flap() {
        this->gravity = this->jump;
    }
    void update() {
        this->gravity += this->velocity;
        this->y += this->gravity;
    }
    bool isDead(float height, const vector<Pipe>& pipes) {
        if (this->y >= height || this->y + this->height <= 0) {
            return true;
        }
        for(auto pipe : pipes) {
            if(!(this->x > pipe.x + pipe.width || this->x+this->width < pipe.x || this->y > pipe.y + pipe.height || this->y + this->height < pipe.y)) {
                return true;
            }
        }
        return false;   
    }
};
class Game {
public:
    Game() {
        cscore = 10000000;
    }
    Game(int cscore_) {
        cscore = cscore_;
    }
	int* start(genome* organisms_, int population_size) {
		organisms = organisms_;
        scores = (int*)malloc(sizeof(int)*population_size);
		pipes.clear();
		birds.clear();
		score = 0;
		spawnInterval = 90;
		interval = 0;
		for(int i=0;i < population_size; i++) {
			birds.push_back(Bird());
		}
		generation++;
		alives = birds.size();
		while(!isItEnd()) {
			update();
		}
		return scores;
	}
	bool isItEnd() {
		for(int i = 0; i < birds.size(); i++) {
			if(birds[i].alive) {
				return false;
			}
		}
		return true;
	}

    void update() {
        backgroundx += backgroundSpeed;
        float nextHoll = 0;
        if (birds.size() > 0) {
            for (int i = 0; i < pipes.size(); i+=2) {
                if (pipes[i].x + pipes[i].width > birds[0].x) {
                    nextHoll = pipes[i].height  / height;
                    break;
                }
            }
        }
        for(int i = 0; i < birds.size(); i++) {
            if(birds[i].alive) {
                float inputs[2];
                inputs[0] = birds[i].y / height;
                inputs[1] =  nextHoll;
                float res = forward(inputs,organisms[i].genes);
                if(res > 0.5) {
                    birds[i].flap();
                }
                birds[i].update();
                if(birds[i].isDead(height, pipes)) {
                    birds[i].alive = false;
                    alives--;
                    // // network score
                    // if(isItEnd()) {
                    //     // infinite loop
                    //     start();
                    // }
                }
				else {
					scores[i]++;
				}
            }
        }
        for(int i = 0; i < pipes.size(); i++) {
            pipes[i].update();
            if(pipes[i].isOut()) {
                pipes.erase(pipes.begin()+i);
                i--;
            }
        }
        if(interval == 0) {
            float deltaBord = 50;
            float pipeHoll = 120;
            float hollPosition = round(((rand()%1000)/1000.)*(height - deltaBord*2-pipeHoll)) + deltaBord;
            Pipe a;
            a.x = width;
            a.y = 0;
            a.height = hollPosition;
            Pipe b;
            b.x = width;
            b.y = hollPosition+pipeHoll;
            b.height = height;
            pipes.push_back(a);
            pipes.push_back(b);
            interval++;
            if(interval == spawnInterval) {
                interval = 0;
            }
            // score++;
            // maxScore = (score > maxScore) ? score : maxScore;

        }
        // cout << "Max Score in this update " << maxScore << endl;
    }
private:
    vector<Pipe> pipes;
    vector<Bird> birds;
    int score = 0;
    int width = 400;
    int height = 800;
    int spawnInterval = 90;
    int interval = 0;
    int alives = 0;
    int generation = 0;
    float backgroundSpeed = 0.5;
    float backgroundx = 0;
	int* scores;
    int maxScore = 0;
    int cscore;
    genome* organisms;
};
// int main() {
// Game gg;
// int MAX_ITERS = 10000;
// // Simulate like FPS
// for(int i = 0; i < MAX_ITERS; i++) {
//     gg.update();
// }
// return 0;
// }