#ifndef UNFOLDING_H
#define UNFOLDING_H

#include <vector>
#include <list>
#include <set>
#include <cstdint> 
#include <stdio.h>
#include <assert.h> 
#include <utility>
#include <queue>
#include <unordered_map>


void mul_2(std::vector<float>& vec);

template <typename TFloat>
void unfold2d(TFloat L, int npart, int ntri, TFloat* pos, TFloat* mass, int64_t* idptr, int64_t* tri);

template <typename TFloat>
void unfold2d_sorted(TFloat L, int npart, int ntri, TFloat* pos, TFloat* mass, int64_t* idptr, int64_t* tri);

template <typename TFloat>
void unfold3d(TFloat L, int npart, int ntets, TFloat* pos, TFloat* mass, int64_t* idptr, int64_t* tets);

template <typename TFloat>
void unfold3d_sorted(TFloat L, int npart, int ntets, TFloat* pos, TFloat* mass, int64_t* idptr, int64_t* tets, int mode);

#endif  // UNFOLDING_H
