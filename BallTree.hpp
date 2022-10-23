#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include <set>
#include <math.h>
#include "BallTree.h"

template <class BALLS>
int BallTree<BALLS>::half (int i, int j, int dir, tr_real_t thr) {
    if (i == (--j)) return i;
    while (true) {
        while (balls->getPos(nr[i],dir) <= thr) if ((++i) == j) return i;
        while (balls->getPos(nr[j],dir) >  thr) if (i == (--j)) return i;
        int tmp = nr[i];
        nr[i] = nr[j];
        nr[j] = tmp;
        if ((++i) == j) return i;
        if (i == (--j)) return i;
    }
}

const int SMAX = 256;
const int threads = 1;
int thread = 0;

template <class BALLS>
tr_addr_t BallTree<BALLS>::build (int ind, int n, int back, int node) {
//    printf("tree build(%d %d %d)\n", ind, n, back);

    int shr[7][SMAX];
    int spt=0;
    shr[3][spt] = ind; shr[4][spt] = n; shr[5][spt] = back; shr[6][spt] = node;
    spt++;
    
    while (spt > 0) {
        spt--;
        ind = shr[3][spt]; n = shr[4][spt]; back = shr[5][spt]; node = shr[6][spt];

        tr_elem elem;
        elem.back = back;
        if (n-ind < 2) {
            elem.flag = 4;
            elem.right = nr[ind];
            elem.a = elem.b = 0;
            tree[node] = elem;
        } else {
            tr_real_t sum=0.0;
            tr_real_t max_span=-1;
            int dir = 0;
            for (int ndir =0; ndir < 3; ndir++) {
                tr_real_t nsum = 0;
                tr_real_t val = balls->getPos(nr[ind],ndir);
                tr_real_t v_min = val, v_max = val;
                for (int i=ind; i<n; i++) {
                    val = balls->getPos(nr[i],ndir);
                    nsum += val;
                    if (val > v_max) v_max = val;
                    if (val < v_min) v_min = val;
                }
                if (v_max - v_min > max_span) {
                    max_span = v_max - v_min;
                    dir = ndir;
                    sum = nsum;
                }
            }
            sum /= (n-ind);
            tr_real_t v_max, v_min, v0, v1;
            int d = half(ind, n, dir, sum);
            assert(ind<d);
            assert(d<n);
            {
                v1 = v_max = balls->getPos(nr[ind],dir) + balls->getRad(nr[ind]);
                v0 = v_min = balls->getPos(nr[n-1],dir) - balls->getRad(nr[n-1]);
                for (int i=ind; i<n; i++) {
                    tr_real_t vala = balls->getPos(nr[i],dir) + balls->getRad(nr[i]);
                    tr_real_t valb = balls->getPos(nr[i],dir) - balls->getRad(nr[i]);
                    if (i < d) {
                        if (vala > v_max) v_max = vala;
                    } else {
                        if (valb < v_min) v_min = valb;
                    }
                }
            }
            elem.right = node+2*(d-ind);
            elem.a = v_min;
            elem.b = v_max;
            elem.flag = dir;
            tree[node] = elem;
            shr[3][spt] = ind; shr[4][spt] = d; shr[5][spt] = node; shr[6][spt] = node+1;
            spt++;
            shr[3][spt] = d; shr[4][spt] = n; shr[5][spt] = back; shr[6][spt] = node + 2*(d-ind);
            spt++;
        }
    }
    return node;
}

template <class BALLS>
bool BallTree<BALLS>::inBall(tr_addr_t ind, tr_real_t* p) {
        tr_real_t r = 0;
        for (int i=0; i<3; i++) {
            tr_real_t d = balls->getPos(ind,i) - p[i];
            r += d*d;
        }
        tr_real_t r2 = balls->getRad(ind);
        return (r < r2*r2);
}

