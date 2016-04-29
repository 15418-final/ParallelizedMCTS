#include <iostream>
#include "CudaGo.h"

int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};

bool CudaBoard::canEat(int i, int j, COLOR color) {
	board[i][j] = color;
	bool result = false;
	COLOR op_color = static_cast<COLOR>(color ^ 3);
	thrust::device_vector<Point*> Q;
	bool visited[BSIZE + 2][BSIZE + 2];
	memset(visited, 0 , (BSIZE + 2) * (BSIZE + 2));
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

bool CudaBoard::isSuicide(int i, int j, COLOR color) {
	thrust::device_vector<Point*> Q;
	bool visited[BSIZE + 2][BSIZE + 2];
	memset(visited, 0 , (BSIZE + 2) * (BSIZE + 2));
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

thrust::device_vector<Point*> CudaBoard::get_next_moves(COLOR color) {
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
//return the number of stones that are killed.
int CudaBoard::update_board(Point* pos, COLOR color) {
	board[pos->i][pos->j] = color;
	//TODO: Remove stones if necessary
	COLOR op_color = static_cast<COLOR>(color ^ 3);
	thrust::device_vector<Point*> Q;
	bool visited[BSIZE + 2][BSIZE + 2];
	memset(visited, 0 , (BSIZE + 2) * (BSIZE + 2));
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
				for (Point* p : temp_stone) {
					board[p->i][p->j] = EMPTY;
				}
			}
			for (Point* p : temp_stone) {
				delete p;
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

bool CudaBoard::EndOfGame() {
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

int CudaBoard::score() {
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