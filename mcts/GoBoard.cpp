#include <iostream>
#include <stdio.h>
#include "GoBoard.h"

bool GoBoard::canEat(int i, int j, COLOR color, Point* point) {
	setBoard(i, j, color);
	bool result = false;
	COLOR op_color = static_cast<COLOR>(color ^ 3);
	q1->clear();
	clearVisited();
	//printf("canEat start\n");
	for (int d = 0 ; d < 4; d++) {
		int ni = i + dir[d][0];
		int nj = j + dir[d][1];
		if (getBoard(ni, nj) == op_color && !isVisited(ni, nj)) {
			q1->push_back(getPoint(point, ni, nj));
			int liberty = 0;
			while (q1->size() != 0) {
				Point f = q1->pop_front();
				setVisited(f.i, f.j, true);
				for (int dd = 0; dd < 4; dd++) {
					ni = f.i + dir[dd][0];
					nj = f.j + dir[dd][1];
					if (isVisited(ni, nj))continue;
					if (getBoard(ni, nj) == op_color) {
						q1->push_back(getPoint(point, ni, nj));
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
	//printf("canEat end\n");
	setBoard(i, j, EMPTY);
	return result;
}

bool GoBoard::isSuicide(int i, int j, COLOR color, Point* point) {

//	printf("isSuicide\n");
	q1->clear();
	clearVisited();
	//printf("isSuicide start\n");
	q1->push_back(getPoint(point, i, j));
	while (q1->size() != 0)  {
		Point f = q1->pop_front();
		setVisited(f.i, f.j, true);
		for (int d = 0 ; d < 4; d++) {
			int ni = f.i + dir[d][0];
			int nj = f.j + dir[d][1];
			if (isVisited(ni, nj))continue;
			if (getBoard(ni, nj) == color) {
				q1->push_back(getPoint(point, ni, nj));
			} else if (getBoard(ni, nj) == EMPTY) {
				return false;
			}
		}
	}
	//		printf("isSuicide done\n");
	return true;
}



std::vector<Point> GoBoard::get_next_moves(Point* point) {
	COLOR color = player;
	std::vector<Point> moves;
	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (getBoard(i, j) == EMPTY) {
				if (!canEat(i, j, color, point) && isSuicide(i, j, color, point)) {
					continue;
				}
				moves.push_back(getPoint(point, i, j));
			}
		}
	}
	return moves;
}

//return the number of stones that are killed.
int GoBoard::update_board(Point pos, Point* point) {
	COLOR color = player;
	setBoard(pos.i, pos.j, color);
	//TODO: Remove stones if necessary
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
			q1->push_back(getPoint(point, ni, nj));
			q2->push_back(q1->front());

			while (q1->size() != 0) {

				Point f = q1->pop_front();
				setVisited(f.i, f.j, true);
				for (int dd = 0; dd < 4; dd++) {
					ni = f.i + dir[dd][0];
					nj = f.j + dir[dd][1];
					if (isVisited(ni, nj))continue;
					if (getBoard(ni, nj) == op_color) {
						//					printf("f3\n");
						Point tp = getPoint(point, ni, nj);
						q1->push_back(tp);
						q2->push_back(tp);
						//					printf("f4\n");
					} else if (getBoard(ni, nj) == EMPTY) {
						liberty++;
					}
				}
			}
			//		printf("f5\n");
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

void GoBoard::print_board() {
	for (int i = 0; i < BSIZE + 1; i++) {
		printf("=");
	}
	printf("\n");
	for (int i = 1; i < BSIZE + 1 ; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (getBoard(i, j) == WHITE) {
				std::cout << "W";
			} else if (getBoard(i, j) == BLACK) {
				std::cout << "B";
			} else {
				std::cout << ".";
			}
		}
		std::cout << std::endl;
	}
	for (int i = 0; i < BSIZE + 1; i++) {
		printf("=");
	}
	printf("\n");
}

int GoBoard::score() {
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