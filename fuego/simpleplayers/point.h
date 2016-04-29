#ifndef POINT_H
#define POINT_H  

#include "SgSystem.h"
#include "SgTimer.h"
#include "SpUtil.h"
#include "SgBlackWhite.h"
class Point{
public:
	int i,j;
	Point(int a, int b):i(a),j(b){}
	Point(SgPoint p){
		i = SgPointUtil::Row(p);
		j = SgPointUtil::Col(p);
	}
	SgPoint ToSgPoint(){
		return SgPointUtil::Pt(j, i);
	}
};

#endif