#ifndef BALLTREE_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include <set>
#include <math.h>

typedef char tr_flag_t;
typedef int tr_addr_t;
typedef double tr_real_t;

struct tr_elem {
    tr_flag_t flag;
    tr_addr_t right;
    tr_addr_t back;
    tr_real_t a;
    tr_real_t b;
};

//template <class BALLS>
//int half_f (BALLS * balls, int* nr, int i, int j, int dir, tr_real_t thr);    
//template <class BALLS>
//void build (BALLS* balls, int* nr, tr_elem* tree, int N);


template <class BALLS>
class BallTree {
private:
    std::vector<tr_elem> tree;
    bool inBall(tr_addr_t i, tr_real_t* p);
    std::vector<tr_addr_t> nr;
    tr_addr_t build (int ind, int n, int back, int node_);
public:
    BALLS* balls;
    inline void Build() {
        tree.clear();
        size_t n = balls->size();
//        printf("Building BallTree (%d) ...\n", n);
        nr.resize(n);
        tree.resize(n*2-1);
        if (n > 0) {
            for (size_t i=0; i<n; ++i) nr[i] = i;
            build(0,n,-1,0);
        }
//        printf("Done (%d)\n", tree.size());
    }
    inline tr_elem* Tree() const { return (tr_elem*) &tree[0]; }
    inline size_t size() const { return tree.size(); }
    inline size_t mem_size() const { return tree.size() * sizeof(tr_elem); }
};

#define BALLTREE_H
#endif
