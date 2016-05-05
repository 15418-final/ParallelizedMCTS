#ifndef POINT_H
#define POINT_H  

#include <cuda.h>
#include <cuda_runtime.h>

class Point{
public:
	int i,j;
	__device__ __host__ Point(int a, int b):i(a),j(b){}

	__device__ __host__ Point():i(0),j(0){}
	
	__device__ __host__ Point(const Point& p){
		i = p.i;
		j = p.j;
	}
};

#endif
