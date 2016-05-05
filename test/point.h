#ifndef POINT_H
#define POINT_H  

class Point{
public:
	int i,j;
	Point(int a, int b):i(a),j(b){}
	
	Point(const Point& p){
		i = p.i;
		j = p.j;
	}
};

#endif