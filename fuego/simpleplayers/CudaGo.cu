#include <iostream>
#include "CudaGo.h"
__device__ __host__ bool CudaBoard::canEat(int i, int j, COLOR color) {
	int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};

	board[i][j] = color;
	bool result = false;
	COLOR op_color = static_cast<COLOR>(color ^ 3);
	thrust::device_vector<Point*> Q;
	clearVisited();
	for (int d = 0 ; d < 4; d++) {
		int ni = i + dir[d][0];
		int nj = j + dir[d][1];
		if (board[ni][nj] == op_color) {
			int liberty = 0;
			Q.push_back(new Point(ni, nj));
			while (Q.size() != 0) {
				Point* f = Q.front();
				Q.erase(Q.begin());
				visited[f->i][f->j] = true;
				for (int dd = 0; dd < 4; dd++) {
					ni = f->i + dir[dd][0];
					nj = f->j + dir[dd][1];
					if (visited[ni][nj] == true)continue;
					if (board[ni][nj] == op_color) {
						Q.push_back(new Point(ni, nj));
					} else if (board[ni][nj] == EMPTY) {
						liberty++;
					} else if (board[ni][nj] == OUT || board[ni][nj] == color) {

					}
				}
				delete f;
			}
			if (liberty == 0) {
				result = true;
				break;
			}
		}
	}

	board[i][j] = EMPTY;
	return result;
}

__device__  __host__ bool CudaBoard::isSuicide(int i, int j, COLOR color) {
	int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};

	thrust::device_vector<Point*> Q;
	clearVisited();
	Q.push_back(new Point(i, j));
	while (Q.size() != 0)  {
		Point* f = Q.front();
		Q.erase(Q.begin());
		visited[f->i][f->j] = true;

		for (int d = 0 ; d < 4; d++) {
			int ni = f->i + dir[d][0];
			int nj = f->j + dir[d][1];			
			if (visited[ni][nj] == true)continue;
			if (board[ni][nj] == color) {
				Q.push_back(new Point(ni, nj));
			} else if (board[ni][nj] == EMPTY) {
				return false;
			}
		}
	}
	return true;
}

__device__  thrust::device_vector<Point*> CudaBoard::get_next_moves_device() {
    COLOR color = currentPlayer;
	thrust::device_vector<Point*> moves;
	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (board[i][j] == EMPTY) { //This is position is empty
				//TODO: Check whether it can eat other stones
				//If not, check whether it's a suicide, which is forbidden.
			//	std::cout << "check" << i << ", " << j <<std::endl;

				if (!canEat(i, j, color) && isSuicide(i, j, color)) {
				//	if (!canEat(i, j, color))std::cout << "can not eat" << std::endl;
				//	if (isSuicide(i, j, color)) std::cout << "is suicide" << std::endl;
					continue;
				}
				moves.push_back(new Point(i, j));
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
			if (board[i][j] == EMPTY) { //This is position is empty
				//TODO: Check whether it can eat other stones
				//If not, check whether it's a suicide, which is forbidden.
			//	std::cout << "check" << i << ", " << j <<std::endl;

				if (!canEat(i, j, color) && isSuicide(i, j, color)) {
				//	if (!canEat(i, j, color))std::cout << "can not eat" << std::endl;
				//	if (isSuicide(i, j, color)) std::cout << "is suicide" << std::endl;
					continue;
				}
				moves.push_back(new Point(i, j));
			}
		}
	}
	return moves;
}

//return the number of stones that are killed.
__device__  int CudaBoard::update_board(Point* pos) {
	int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};

	COLOR color = currentPlayer;
	board[pos->i][pos->j] = color;
	//TODO: Remove stones if necessary
	COLOR op_color = static_cast<COLOR>(color ^ 3);
	thrust::device_vector<Point*> Q;
	clearVisited();
	thrust::device_vector<Point*> temp_stone;
	int total = 0;
	for (int d = 0 ; d < 4; d++) {
		int ni = pos->i + dir[d][0];
		int nj = pos->j + dir[d][1];
		if (board[ni][nj] == op_color) {
			int liberty = 0;
			Q.push_back(new Point(ni, nj));
			temp_stone.push_back(Q.front());
			while (Q.size() != 0) {
				Point* f = Q.front();
			    Q.erase(Q.begin());
				visited[f->i][f->j] = true;
				for (int dd = 0; dd < 4; dd++) {
					ni = f->i + dir[dd][0];
					nj = f->j + dir[dd][1];
					if (visited[ni][nj] == true)continue;
					if (board[ni][nj] == op_color) {
						Point* tp = new Point(ni, nj);
						Q.push_back(tp);
						temp_stone.push_back(tp);
					} else if (board[ni][nj] == EMPTY) {
						liberty++;
					} else if (board[ni][nj] == OUT || board[ni][nj] == color) {

					}
				}
			}
			if (liberty == 0) {
				total += temp_stone.size();
				for (thrust::device_vector<Point*>::iterator it = temp_stone.begin(); it != temp_stone.end(); it++) {
					Point* p = *it;
					board[p->i][p->j] = EMPTY;
				}
			}
			for (thrust::device_vector<Point*>::iterator it = temp_stone.begin(); it != temp_stone.end(); it++) {
				delete *it;
			}
			temp_stone.clear();
		}
	}

	currentPlayer = static_cast<COLOR>(color ^ 3);

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

	return white - black;
}