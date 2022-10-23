#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include <set>
#include <math.h>
#include "BallTree.h"
#include <cub/cub.cuh>

template <class BALLS>
__host__ __device__ int half_f (BALLS* balls, int* nr, int i, int j, int dir, tr_real_t thr) {
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
void build (BALLS* balls, int* nr, tr_elem* tree, int N) {
//    printf("tree build(%d %d %d)\n", ind, n, back);

    int shr[4][SMAX];
    int spt=0;
    
    for (size_t i=0; i<N; ++i) nr[i] = i;
      
    shr[0][spt] = 0; shr[1][spt] = N; shr[2][spt] = -1; shr[3][spt] = 0;
    spt++;
    
    while (spt > 0) {
        spt--;
        int ind = shr[0][spt];
        int n = shr[1][spt];
        int back = shr[2][spt];
        int node = shr[3][spt];

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
            int d = half_f(balls, nr, ind, n, dir, sum);
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
            shr[0][spt] = ind; shr[1][spt] = d; shr[2][spt] = node; shr[3][spt] = node+1;
            spt++;
            shr[0][spt] = d; shr[1][spt] = n; shr[2][spt] = back; shr[3][spt] = node + 2*(d-ind);
            spt++;
        }
    }
}

const int BUILD_BLOCK_SIZE = 1;

template <class BALLS>
__global__ void buildgpu (BALLS* balls, int* nr, tr_elem* tree, int N) {
//    printf("tree build(%d %d %d)\n", ind, n, back);
    int thi = threadIdx.x;
    int thn = blockDim.x;
//    typedef cub::BlockScan<int, BUILD_BLOCK_SIZE> BlockScan;
//    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int shr[4][SMAX];
    int spt = 1;
    __shared__ int tleaf;
    for (size_t i=thi; i<N; i+=thn) nr[i] = i;
    
    if (thi == 0) {
        shr[0][spt] = 0; shr[1][spt] = N; shr[2][spt] = -1; shr[3][spt] = 0;
    }
        
    while (spt > 0) {
        __syncthreads();
        int ind, n, back, node;
        if (thi < spt) {
            int spi = spt - 1 - thi;
            ind  = shr[0][spi];
            n    = shr[1][spi];
            back = shr[2][spi];
            node = shr[3][spi];
        } else {
            n = ind = 0;
        }
        int leaf = (n-ind < 2) ? 0 : 1;
        int cleaf;
//        BlockScan(temp_storage).ExclusiveSum(leaf, cleaf);
        cleaf = 0;
        if (thi == thn-1) {
            tleaf = cleaf + leaf;
        }
        if (thn < spt) {
            spt = spt - thn;
        } else {
            spt = 0;
        }
        __syncthreads();
        int n1 = spt + cleaf;
        int n2 = n1 + tleaf;
        spt = spt + tleaf + tleaf;

        if (n == 0) {
        if (n-ind < 2) {
            tr_elem elem;
            elem.back = back;
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
            int d = half_f(balls, nr, ind, n, dir, sum);
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
        tr_elem elem;
        elem.back = back;
            elem.right = node+2*(d-ind);
            elem.a = v_min;
            elem.b = v_max;
            elem.flag = dir;
            tree[node] = elem;
            shr[0][n1] = ind; shr[1][n1] = d; shr[2][n1] = node; shr[3][n1] = node+1;
            shr[0][n2] = d; shr[1][n2] = n; shr[2][n2] = back; shr[3][n2] = node + 2*(d-ind);
        }
        }
    }
}


template <class BALLS>
tr_addr_t BallTree<BALLS>::build (int ind, int n, int back, int node) {
//    printf("tree build(%d %d %d)\n", ind, n, back);

    int shr[4][SMAX];
    int spt=0;
    shr[0][spt] = ind; shr[1][spt] = n; shr[2][spt] = back; shr[3][spt] = node;
    spt++;
    
    while (spt > 0) {
        spt--;
        ind = shr[0][spt]; n = shr[1][spt]; back = shr[2][spt]; node = shr[3][spt];

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
            int d = half_f(balls, &nr[0], ind, n, dir, sum);
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
            shr[0][spt] = ind; shr[1][spt] = d; shr[2][spt] = node; shr[3][spt] = node+1;
            spt++;
            shr[0][spt] = d; shr[1][spt] = n; shr[2][spt] = back; shr[3][spt] = node + 2*(d-ind);
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
