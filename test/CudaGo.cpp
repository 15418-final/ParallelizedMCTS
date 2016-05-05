#include <iostream>
#include <stdio.h>
#include "CudaGo.h"

bool CudaBoard::canEat(int i, int j, COLOR color) {
	int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};
	setBoard(i, j, color);
	bool result = false;
	COLOR op_color = static_cast<COLOR>(color ^ 3);
	Deque<Point*>* Q = new Deque<Point*>();
	clearVisited();
	//printf("canEat start\n");
	for (int d = 0 ; d < 4; d++) {
		int ni = i + dir[d][0];
		int nj = j + dir[d][1];
		if (getBoard(ni, nj) == op_color && !isVisited(ni, nj)) {
			Q->push_back(new Point(ni, nj));
			int liberty = 0;
			while (Q->size() != 0) {
				Point* f = Q->pop_front();
				setVisited(f->i, f->j, true);
				for (int dd = 0; dd < 4; dd++) {
					ni = f->i + dir[dd][0];
					nj = f->j + dir[dd][1];
					if (isVisited(ni, nj))continue;
					if (getBoard(ni, nj) == op_color) {
						Q->push_back(new Point(ni, nj));
					} else if (getBoard(ni, nj) == EMPTY) {
						liberty++;
					} 
				}
				delete f;
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
	delete Q;
	return result;
}

bool CudaBoard::isSuicide(int i, int j, COLOR color) {
	int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};
//	printf("isSuicide\n");
	Deque<Point*>* Q = new Deque<Point*>();
	clearVisited();
	//printf("isSuicide start\n");
	Q->push_back(new Point(i, j));
	while (Q->size() != 0)  {
		Point* f = Q->pop_front();
		setVisited(f->i, f->j, true);
		for (int d = 0 ; d < 4; d++) {
			int ni = f->i + dir[d][0];
			int nj = f->j + dir[d][1];			
			if (isVisited(ni, nj))continue;
			if (getBoard(ni, nj) == color) {
				Q->push_back(new Point(ni, nj));
			} else if (getBoard(ni, nj) == EMPTY) {
				cleanQueue(Q);
				delete Q;
				return false;
			}
		}
		delete f;
	}
	//		printf("isSuicide done\n");
	delete Q;
	return true;
}

Deque<Point*>* CudaBoard::get_next_moves() {
    COLOR color = player;
	Deque<Point*>* moves = new Deque<Point*>();
	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (getBoard(i, j) == EMPTY) { //This is position is empty
				if (!canEat(i, j, color) && isSuicide(i, j, color)) {
					continue;
				}
				moves->push_back(new Point(i, j));
			}
		}
	}
	return moves;
}

//return the number of stones that are killed.
int CudaBoard::update_board(Point* pos) {
	int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};
//printf("update_board\n");
	COLOR color = player;
	setBoard(pos->i, pos->j, color);
	//TODO: Remove stones if necessary
	COLOR op_color = static_cast<COLOR>(color ^ 3);

	Deque<Point*>* Q = new Deque<Point*>();
	Deque<Point*>* temp_stone = new Deque<Point*>();

	clearVisited();
//	printf("ready to play\n");
	int total = 0;
	for (int d = 0 ; d < 4; d++) {
	//	printf("direction:%d\n", d);
		int ni = pos->i + dir[d][0];
		int nj = pos->j + dir[d][1];
		if (getBoard(ni, nj) == op_color && !isVisited(ni, nj)) {
			int liberty = 0;
			Q->push_back(new Point(ni, nj));
			temp_stone->push_back(Q->front());
	//		printf("Q size:%d, temp_stone size:%d\n",Q->size(), temp_stone->size());
			while (Q->size() != 0) {
	//			printf("Q size:%d, temp_stone size:%d\n",Q->size(), temp_stone->size());
				Point* f = Q->pop_front();
				setVisited(f->i, f->j, true);
				for (int dd = 0; dd < 4; dd++) {
					ni = f->i + dir[dd][0];
					nj = f->j + dir[dd][1];
					if (isVisited(ni, nj))continue;
					if (getBoard(ni, nj) == op_color) {
	//					printf("f3\n");
						Point* tp = new Point(ni, nj);
						Q->push_back(tp);
						temp_stone->push_back(tp);
	//					printf("f4\n");
					} else if (getBoard(ni, nj) == EMPTY) {
						liberty++;
					}
				}
			}
	//		printf("f5\n");
			if (liberty == 0) {
				total += temp_stone->size();
				for (Deque<Point*>::iterator it = temp_stone->begin(); it != temp_stone->end(); it++) {
					Point* p = *it;
					setBoard(p->i, p->j, EMPTY);
					delete p;
				}
			}else{
				for (Deque<Point*>::iterator it = temp_stone->begin(); it != temp_stone->end(); it++) {
					delete *it;
				}
			}
			temp_stone->clear();
		}
	}
	delete Q;
	delete temp_stone;
	player = op_color;

	return total;
}

void CudaBoard::print_board() {
	for(int i = 0; i < BSIZE + 1;i++) {
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
				std::cout<<".";
			}
		}
		std::cout << std::endl;
	}
	for(int i = 0; i < BSIZE + 1;i++) {
		printf("=");
	}
	printf("\n");
}

bool CudaBoard::EndOfGame() {
	COLOR color = player;
	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (getBoard(i, j) == EMPTY) {
				if (!canEat(i, j, color) && isSuicide(i, j, color)) {
					continue;
				}
				return false;
			}
		}
	}
	return true;
}

int CudaBoard::score() {
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

void CudaBoard::cleanQueue(Deque<Point*>* queue) {
	for (Deque<Point*>::iterator it = queue->begin(); it != queue->end(); it++) {
		Point* p = *it;
		delete p;
	}
}