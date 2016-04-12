#include <iostream>
#include "go.h"
#include <vector>
#include <queue>

int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};

bool Board::canEat(int i, int j, COLOR color) {
	COLOR op_color = static_cast<COLOR>(color ^ 3);
	std::queue<Pair*> Q;
	bool visited[BSIZE + 2][BSIZE + 2];
	memset(visited, 0 , (BSIZE + 2) * (BSIZE + 2));
	for (int d = 0 ; d < 4; d++) {
		int ni = i + dir[d][0];
		int nj = j + dir[d][1];
		if (board[ni][nj] == op_color) {
			int liberty = 0;
			Q.push(new Pair(ni, nj));
			while (!Q.empty()) {
				Pair* f = Q.front();
				Q.pop();
				visited[f->i][f->j] = true;
				for (int dd = 0; dd < 4; dd++) {
					ni = f->i + dir[dd][0];
					nj = f->j + dir[dd][1];
					if (visited[ni][nj] == true)continue;
					if (board[ni][nj] == op_color) {
						Q.push(new Pair(ni, nj));
					} else if (board[ni][nj] == EMPTY) {
						liberty++;
					} else if (board[ni][nj] == OUT || board[ni][nj] == color) {

					}
				}
				delete f;
			}
			if (liberty == 0)return true;
		}
	}
	return false;
}

bool Board::isSuicide(int i, int j, COLOR color) {
	std::queue<Pair*> Q;
	bool visited[BSIZE + 2][BSIZE + 2];
	memset(visited, 0 , (BSIZE + 2) * (BSIZE + 2));
	Q.push(new Pair(i, j));
	while (!Q.empty()) {
		Pair* f = Q.front();
		Q.pop();
		visited[f->i][f->j] = true;
		for (int d = 0 ; d < 4; d++) {
			int ni = i + dir[d][0];
			int nj = j + dir[d][1];
			if (visited[ni][nj] == true)continue;
			if (board[ni][nj] == color) {
				Q.push(new Pair(ni, nj));
			} else if (board[ni][nj] == EMPTY) {
				return false;
			}
		}
	}
	return true;
}

std::vector<Pair*> Board::get_next_moves(COLOR color) {
	std::vector<Pair*> moves;
	for (int i = 1; i < BSIZE + 1; i++) {
		for (int j = 1; j < BSIZE + 1; j++) {
			if (board[i][j] == EMPTY) { //This is position is empty
				//TODO: Check whether it can eat other stones
				//If not, check whether it's a suicide, which is forbidden.
				std::cout << "check" << std::endl;

				if (!canEat(i, j, color) && isSuicide(i, j, color)) {
					if (!canEat(i, j, color))std::cout << "can not eat" << std::endl;
					if (isSuicide(i, j, color)) std::cout << "is suicide" << std::endl;
					continue;
				}
				moves.push_back(new Pair(i, j));
			}
		}
	}
	return moves;
}
//return the number of stones that are killed.
int Board::update_board(Pair* pos, COLOR color) {
	board[pos->i][pos->j] = color;
	//TODO: Remove stones if necessary
	COLOR op_color = static_cast<COLOR>(color ^ 3);
	std::queue<Pair*> Q;
	bool visited[BSIZE + 2][BSIZE + 2];
	memset(visited, 0 , (BSIZE + 2) * (BSIZE + 2));
	std::vector<Pair*> temp_stone;
	int total = 0;
	for (int d = 0 ; d < 4; d++) {
		int ni = pos->i + dir[d][0];
		int nj = pos->j + dir[d][1];
		if (board[ni][nj] == op_color) {
			int liberty = 0;
			Q.push(new Pair(ni, nj));
			temp_stone.push_back(Q.front());
			while (!Q.empty()) {
				Pair* f = Q.front();
				Q.pop();
				visited[f->i][f->j] = true;
				for (int dd = 0; dd < 4; dd++) {
					ni = f->i + dir[dd][0];
					nj = f->j + dir[dd][1];
					if (visited[ni][nj] == true)continue;
					if (board[ni][nj] == op_color) {
						Pair* tp = new Pair(ni, nj);
						Q.push(tp);
						temp_stone.push_back(tp);
					} else if (board[ni][nj] == EMPTY) {
						liberty++;
					} else if (board[ni][nj] == OUT || board[ni][nj] == color) {

					}
				}
			}
			if (liberty == 0) {
				total += temp_stone.size();
				for (Pair* p : temp_stone) {
					board[p->i][p->j] = EMPTY;
				}
			}
			for (Pair* p : temp_stone) {
				delete p;
			}
		}
	}
	return total;
}

void Board::print_board() {
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

COLOR Board::find_winner() {
	return WHITE;
}