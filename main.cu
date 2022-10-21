#include "BallTree.h"
#include "BallTree.hpp"
#include <vector>
#include <random>

struct Balls {
    int n;
    std::vector <double> pos[3];
    std::vector<double> rad;
    inline long size() {return n; }
    inline double getPos(int i, int j) { return pos[j][i]; }
    inline double getRad(int i) { return rad[i]; }
    Balls(int n_) : n(n_) { for (int j=0;j<3;j++) pos[j].resize(n); rad.resize(n); } 
};


int main() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    int n = 1000000;
    Balls balls(n);
    std::uniform_real_distribution<double> pos_dist(0, 1);
    std::uniform_real_distribution<double> rad_dist(0.01, 0.1);
    std::default_random_engine random_engine;
    for (int i=0;i<n;i++) {
        for (int j=0;j<3;j++) balls.pos[j][i] = pos_dist(random_engine);
        balls.rad[i] = rad_dist(random_engine);
    }
    BallTree<Balls> tree;
    tree.balls = &balls;
	printf("building ...\n");
	cudaEventRecord(start);
    tree.Build();
	cudaEventRecord(stop);
	printf("done\n");

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time:%f\n", milliseconds);
    return 0;
}
