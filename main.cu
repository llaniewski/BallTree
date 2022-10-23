#include <vector>
#include <random>
#include <cuda.h>

#include "BallTree.h"
#include "BallTree.hpp"

namespace old {
	#undef BALLTREE_H
	#include "old/BallTree.h"
	#include "old/BallTree.hpp"
};

struct ball {
	double pos[3];
	double rad;
};

struct Balls {
	int n;
	ball* balls;
	inline long size() { return n; }
	inline __host__ __device__ double getPos(int i, int j) { return balls[i].pos[j]; }
	inline __host__ __device__ double getRad(int i) { return balls[i].rad; }
};


int main() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    int n = 100000;
    Balls balls;
    std::uniform_real_distribution<double> pos_dist(0, 1);
    std::uniform_real_distribution<double> rad_dist(0.01, 0.1);
    std::default_random_engine random_engine;
	balls.balls = new ball[n];
	balls.n = n;
    for (int i=0;i<n;i++) {
        for (int j=0;j<3;j++) balls.balls[i].pos[j] = pos_dist(random_engine);
        balls.balls[i].rad = rad_dist(random_engine);
    }
    old::BallTree<Balls> oldtree;
    oldtree.balls = &balls;
	printf("building old ...\n");
    oldtree.Build();
	printf("done\n");

//    BallTree<Balls> tree;
//    tree.balls = &balls;

        size_t N = balls.size();
        int* nr = new int[N];
	size_t M = N*2-1;
        tr_elem* tree = new tr_elem[N*2-1];

	int * gnr; tr_elem* gtree; ball* gballs; Balls* ggballs;
	cudaMalloc( &gnr,   sizeof(int) * N);
	cudaMalloc( &gtree, sizeof(tr_elem) * M);
	cudaMalloc( &gballs, sizeof(ball) * N);
	cudaMalloc( &ggballs, sizeof(Balls));
	{
		Balls cgballs;
		cgballs.balls = gballs;
		cgballs.n = balls.n;
		cudaMemcpy( gballs, balls.balls, sizeof(ball)*N, cudaMemcpyHostToDevice  );
		cudaMemcpy( ggballs, &cgballs, sizeof(Balls), cudaMemcpyHostToDevice  );
	}
	printf("building ...\n");
	cudaEventRecord(start);
	build( &balls, nr, tree, N); 
	cudaEventRecord(stop);
	printf("done\n");

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("time:%f\n", milliseconds);

	printf("building ...\n");
	cudaEventRecord(start);
	buildgpu<<< 1, 1 >>>( ggballs, gnr, gtree, N); 
	cudaEventRecord(stop);
	printf("done\n");

	cudaEventSynchronize(stop);
//	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("time:%f\n", milliseconds);

	cudaMemcpy( tree, gtree, sizeof(tr_elem)*M, cudaMemcpyDeviceToHost );

	if (oldtree.size() != M ) {
		printf("Wrong sizes: old:%ld new:%ld\n", oldtree.size(), M);
		return -1;
	}
	for (int i=0; i<M; i++) {
		tr_elem el1 = tree[i];
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
