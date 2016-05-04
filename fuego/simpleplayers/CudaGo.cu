#include <iostream>
#include <stdio.h>
#include "CudaGo.h"

__device__ __host__ bool CudaBoard::canEat(int i, int j, COLOR color) {
	int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};

	board[i][j] = color;
	bool result = false;
	COLOR op_color = static_cast<COLOR>(color ^ 3);
	Deque<Point*>* Q = new Deque<Point*>();
	clearVisited();
	for (int d = 0 ; d < 4; d++) {
		int ni = i + dir[d][0];
		int nj = j + dir[d][1];

		if (board[ni][nj] == op_color && !visited[ni][nj]) {
			// printf("ready to push in suicide:\n");
			Q->push_back(new Point(ni, nj));
			// printf("wtf\n");
			int liberty = 0;
			while (Q->size() != 0) {
				// printf("will pop front\n");
				// Q->pop_front();
				// printf("test test2\n");
				Point* f = Q->pop_front();
				// printf("test test3\n");
				visited[f->i][f->j] = true;
				// printf("??1\n");
				for (int dd = 0; dd < 4; dd++) {
					ni = f->i + dir[dd][0];
					nj = f->j + dir[dd][1];
					// printf("??2\n");
					if (visited[ni][nj] == true)continue;
					if (board[ni][nj] == op_color) {
						// printf("will push_back\n");
						Q->push_back(new Point(ni, nj));
						// printf("after push_back\n");
					} else if (board[ni][nj] == EMPTY) {
						liberty++;
						// printf("??3\n");
					} else if (board[ni][nj] == OUT || board[ni][nj] == color) {
						// printf("??4\n");
					}
				}
				if (liberty == 0) {
					result = true;
					break;
				}
				// printf("want to delete f\n");
				delete f;
				// printf("after delte f\n");
			}
			// printf("dir %d done\n",d);
		}
	}
	board[i][j] = EMPTY;
	// printf("want to delete Q\n");
	delete Q;
	// printf("suicide ends with:%d\n",result);
	return result;
}

__device__  __host__ bool CudaBoard::isSuicide(int i, int j, COLOR color) {
	int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};

	Deque<Point*>* Q = new Deque<Point*>();
	// printf("isSuicide: f0\n");
	clearVisited();
	// printf("trying to push: %d, %d\n", i, j);
	// printf("Q size:%d\n",Q->size());
	Q->push_back(new Point(i, j));

//	printf("before loop\n");
	while (Q->size() != 0)  {
		Point* f = Q->pop_front();
		// printf("isSuicide: f2\n");
		visited[f->i][f->j] = true;

		for (int d = 0 ; d < 4; d++) {
			int ni = f->i + dir[d][0];
			int nj = f->j + dir[d][1];		
			// printf("isSuicide: f4\n");
			if (visited[ni][nj] == true)continue;
			if (board[ni][nj] == color) {
				Q->push_back(new Point(ni, nj));
				// printf("isSuicide: f5\n");
			} else if (board[ni][nj] == EMPTY) {

				cleanQueue(Q);
				delete Q;
		//		printf("isSuicide done\n");
				return false;
			}
		}
		delete f;
	}
	delete Q;
	return true;
}

__device__  Deque<Point*>* CudaBoard::get_next_moves_device() {

    COLOR color = currentPlayer;
	Deque<Point*>* moves = new Deque<Point*>();
	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (board[i][j] == EMPTY) { //This is position is empty
				//TODO: Check whether it can eat other stones
				//If not, check whether it's a suicide, which is forbidden.

				// printf("checkpoint 0\n");
				if (!canEat(i, j, color) && isSuicide(i, j, color)) {
				//	if (!canEat(i, j, color))std::cout << "can not eat" << std::endl;
				//	if (isSuicide(i, j, color)) std::cout << "is suicide" << std::endl;
					continue;
				}
				// printf("checkpoint 1\n");
				moves->push_back(new Point(i, j));
				// printf("checkpoint 2\n");
				// printf("next move size:%d\n",moves->size());
			}
		}
	}
	return moves;
}

std::vector<Point*> CudaBoard::get_next_moves_host() {

    COLOR color = currentPlayer;
	std::vector<Point*> moves;
	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (board[i][j] == EMPTY) {
				if (!canEat(i, j, color) && isSuicide(i, j, color)) {
					continue;
				}
				moves.push_back(new Point(i, j));
			}
		}
	}
	return moves;
}

//return the number of stones that are killed.
__device__  __host__ int CudaBoard::update_board(Point* pos) {
	int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};
//printf("update_board\n");
	COLOR color = currentPlayer;
	board[pos->i][pos->j] = color;
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
		if (board[ni][nj] == op_color && !visited[ni][nj]) {
			int liberty = 0;
			Q->push_back(new Point(ni, nj));
			temp_stone->push_back(Q->front());
	//		printf("Q size:%d, temp_stone size:%d\n",Q->size(), temp_stone->size());
			while (Q->size() != 0) {
	//			printf("Q size:%d, temp_stone size:%d\n",Q->size(), temp_stone->size());
				Point* f = Q->pop_front();
	//			printf("f1\n");
				visited[f->i][f->j] = true;
	//			printf("f2\n");
				for (int dd = 0; dd < 4; dd++) {
					ni = f->i + dir[dd][0];
					nj = f->j + dir[dd][1];
					if (visited[ni][nj] == true)continue;
					if (board[ni][nj] == op_color) {
	//					printf("f3\n");
						Point* tp = new Point(ni, nj);
						Q->push_back(tp);
						temp_stone->push_back(tp);
	//					printf("f4\n");
					} else if (board[ni][nj] == EMPTY) {
						liberty++;
					} else if (board[ni][nj] == OUT || board[ni][nj] == color) {

					}
				}
			}
	//		printf("f5\n");
			if (liberty == 0) {
				total += temp_stone->size();
				for (Deque<Point*>::iterator it = temp_stone->begin(); it != temp_stone->end(); it++) {
					Point* p = *it;
					board[p->i][p->j] = EMPTY;
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
//	printf("play done\n");
	currentPlayer = static_cast<COLOR>(color ^ 3);
	// for (int i = 1; i < BSIZE + 1 ; i++) {
	// 	for (int j = 1; j < BSIZE + 1; j++) {
	// 		if (board[i][j] == WHITE) {
	// 			printf("W");
	// 		} else if (board[i][j] == BLACK) {
	// 			printf("B");
	// 		} else {
	// 			printf(".");
	// 		}
	// 	}
	// 	printf("\n");
	// }
	return total;
}

void CudaBoard::print_board() {
	for (int i = 1; i < BSIZE + 1 ; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (board[i][j] == WHITE) {
				std::cout << "W";
			} else if (board[i][j] == BLACK) {
				std::cout << "B";
			} else {
				std::cout << ".";
			}
		}
		std::cout << std::endl;
	}
}

__device__ bool CudaBoard::EndOfGame() {
	COLOR color = currentPlayer;
	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (board[i][j] == EMPTY) {
				if (!canEat(i, j, color) && isSuicide(i, j, color)) {
					continue;
				}
				return false;
			}
		}
	}
	return true;
}

__device__  int CudaBoard::score() {
	int black = 0;
	int white = 0;

	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (board[i][j] == WHITE) {
				white++;
			} else if (board[i][j] == BLACK) {
				black++;
			}
		}
	}

	return black - white;
}

__device__ __host__ void CudaBoard::cleanQueue(Deque<Point*>* queue) {
	for (Deque<Point*>::iterator it = queue->begin(); it != queue->end(); it++) {
		Point* p = *it;
		delete p;
	}
}