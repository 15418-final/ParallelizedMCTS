#ifndef POINT_H
#define POINT_H  

#include "SgSystem.h"
#include "SgTimer.h"
#include "SpUtil.h"
#include "SgBlackWhite.h"

#include <cuda.h>
#include <cuda_runtime.h>

class Point{
public:
	int i,j;
	__device__ __host__ Point(int a, int b):i(a),j(b){}
	__device__ __host__ Point(){}

	Point(const Point& p){
		i = p.i;
		j = p.j;
	}

	Point(SgPoint& p){
		i = SgPointUtil::Row(p);
		j = SgPointUtil::Col(p);
	}

	SgPoint ToSgPoint(){
		return SgPointUtil::Pt(j, i);
	}
};

#endif