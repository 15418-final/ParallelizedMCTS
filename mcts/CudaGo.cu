#include <iostream>
#include <stdio.h>
#include "CudaGo.h"

__device__ __host__ bool CudaBoard::canEat(int i, int j, COLOR color) {
	setBoard(i, j, color);
	bool result = false;
	COLOR op_color = static_cast<COLOR>(color ^ 3);
	q1->clear();
	clearVisited();

	for (int d = 0 ; d < 4; d++) {
		int ni = i + dir[d][0];
		int nj = j + dir[d][1];
		if (getBoard(ni, nj) == op_color && !isVisited(ni, nj)) {
			q1->push_back(Point(ni, nj));
			int liberty = 0;
			while (q1->size() != 0) {
				Point f = q1->pop_front();
				setVisited(f.i, f.j, true);
				for (int dd = 0; dd < 4; dd++) {
					ni = f.i + dir[dd][0];
					nj = f.j + dir[dd][1];
					if (isVisited(ni, nj))continue;
					if (getBoard(ni, nj) == op_color) {
						q1->push_back(Point(ni, nj));
					} else if (getBoard(ni, nj) == EMPTY) {
						liberty++;
					} 
				}
			}
			
			if (liberty == 0) {
				result = true;
				break;
			}
		}
		if (result) break;
	}

	setBoard(i, j, EMPTY);
	return result;
}

__device__  __host__ bool CudaBoard::isSuicide(int i, int j, COLOR color) {
	q1->clear();
	clearVisited();

	q1->push_back(Point(i, j));
	while (q1->size() != 0)  {
		Point f = q1->pop_front();
		setVisited(f.i, f.j, true);
		for (int d = 0 ; d < 4; d++) {
			int ni = f.i + dir[d][0];
			int nj = f.j + dir[d][1];			
			if (isVisited(ni, nj))continue;
			if (getBoard(ni, nj) == color) {
				q1->push_back(Point(ni, nj));
			} else if (getBoard(ni, nj) == EMPTY) {
				return false;
			}
		}
	}

	return true;
}

__device__  Point CudaBoard::get_next_moves_device(double seed) {
	COLOR color = player;
	int step = seed * remain;
	int current = 0;
	q2->clear();
	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (getBoard(i, j) == EMPTY) { 
				if (!canEat(i, j, color) && isSuicide(i, j, color)) {
					continue;
				}

				q2->push_back(Point(i, j));
				if (current == step) return Point(i, j);
				current++;
			}
		}
	}

	if (current != 0) return (*q2)[(int)(seed*q2->size())];
	return Point(-1,-1);
}

std::vector<Point> CudaBoard::get_next_moves_host() {
	COLOR color = player;
	std::vector<Point> moves;
	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (getBoard(i, j) == EMPTY) {
				if (!canEat(i, j, color) && isSuicide(i, j, color)) {
					continue;
				}
				moves.push_back(Point(i, j));
			}
		}
	}
	return moves;
}

//return the number of stones that are killed.
__device__  __host__ int CudaBoard::update_board(Point pos) {
	COLOR color = player;
	setBoard(pos.i, pos.j, color);

	COLOR op_color = static_cast<COLOR>(color ^ 3);

	q1->clear();
	q2->clear(); // temp_stone

	clearVisited();
	int total = 0;
	for (int d = 0 ; d < 4; d++) {
		int ni = pos.i + dir[d][0];
		int nj = pos.j + dir[d][1];
		if (getBoard(ni, nj) == op_color && !isVisited(ni, nj)) {
			int liberty = 0;
			q1->push_back(Point(ni, nj));
			q2->push_back(q1->front());
			while (q1->size() != 0) {
				Point f = q1->pop_front();
				setVisited(f.i, f.j, true);
				for (int dd = 0; dd < 4; dd++) {
					ni = f.i + dir[dd][0];
					nj = f.j + dir[dd][1];
					if (isVisited(ni, nj))continue;
					if (getBoard(ni, nj) == op_color) {
						Point tp = Point(ni, nj);
						q1->push_back(tp);
						q2->push_back(tp);
					} else if (getBoard(ni, nj) == EMPTY) {
						liberty++;
					}
				}
			}
			
			if (liberty == 0) {
				total += q2->size();
				for (Deque<Point>::iterator it = q2->begin(); it != q2->end(); it++) {
					Point p = *it;
					setBoard(p.i, p.j, EMPTY);
				}
			}
			q2->clear();
		}
	}
	player = op_color;

	return total;
}

__device__ __host__ void CudaBoard::print_board() {
	for (int i = 0; i < BSIZE + 1; i++) {
		printf("=");
	}
	printf("\n");
	for (int i = 1; i < BSIZE + 1 ; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (getBoard(i, j) == WHITE) {
				printf("W");
			} else if (getBoard(i, j) == BLACK) {
				printf("B");
			} else {
				printf(".");
			}
		}
		printf("\n");
	}
	for (int i = 0; i < BSIZE + 1; i++) {
		printf("=");
	}
	printf("\n");
}

__device__  __host__ int CudaBoard::score() {
	int black = 0;
	int white = 0;

	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (getBoard(i, j) == WHITE) {
				white++;
			} else if (getBoard(i, j) == BLACK) {
				black++;
			}
		}
	}

	return black - white;
}