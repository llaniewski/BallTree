#include <vector>
#include <random>

	#include "BallTree.h"
	#include "BallTree.hpp"
namespace old {
	#undef BALLTREE_H
	#include "old/BallTree.h"
	#include "old/BallTree.hpp"
};

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
    old::BallTree<Balls> oldtree;
    oldtree.balls = &balls;
	printf("building old ...\n");
    oldtree.Build();
	printf("done\n");

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

	if (oldtree.size() != tree.size() ) {
		printf("Wrong sizes: old:%ld new:%ld\n", oldtree.size(), tree.size());
		return -1;
	}
	for (int i=0; i<oldtree.size(); i++) {
		tr_elem el1 = tree.Tree()[i];
		old::tr_elem el2 = oldtree.Tree()[i];
		bool sel = true;
		sel &= el1.flag  == el2.flag;
		sel &= el1.right == el2.right;
		sel &= el1.back  == el2.back;
		sel &= el1.a     == el2.a;
		sel &= el1.b     == el2.b;

		if (!sel) {
			printf("======================== Wrong ! ========================\n");
			printf(": old:%d new:%d\n", (int) el1.flag   , (int) el2.flag  );
			printf(": old:%d new:%d\n", (int)  el1.right , (int) el2.right  );
			printf(": old:%d new:%d\n", (int)  el1.back  , (int) el2.back  );
			printf(": old:%lf new:%lf diff: %lf\n", el1.a     , el2.a, el2.a - el1.a );
			printf(": old:%lf new:%lf diff: %lf\n", el1.b     , el2.b, el2.b - el1.b );
			printf("======================== Wrong ! ========================\n");
			return -1;
		}
	}
	
    return 0;
}
