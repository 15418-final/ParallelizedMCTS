#ifndef GO_H
#define GO_H

#include <vector>
#include "utils.h"

//std::unordered_set<X, MyHash> s;

const int BSIZE = 9;
enum COLOR {WHITE = 1, BLACK = 2, EMPTY = 0, OUT = 3};

class Board {
private:
	int **board;
	bool canEat(int i, int j, COLOR color);
	bool isSuicide(int i, int j, COLOR color);
public:
	Board() {
		board = new int*[BSIZE + 2];
		for (int i = 0; i < BSIZE + 2; i++) {
			board[i] = new int[BSIZE + 2];
			memset(board[i], 0, sizeof(int) * (BSIZE + 2));
		}
		//set the border
		for (int i = 0; i < BSIZE + 2; i++) {
			board[i][0] = 3;
			board[0][i] = 3;
			board[i][BSIZE + 1] = 3;
			board[BSIZE + 1][i] = 3;
		}
	}

	//copy constructor
	Board(Board& b) {
		board = new int*[BSIZE + 2];
		for (int i = 0; i < BSIZE + 2; i++) {
			board[i] = new int[BSIZE + 2];
			memcpy(&board[i], &b.board[i], sizeof(int) * (BSIZE + 2));
		}

	}
	~Board() {
		for (int i = 0; i < BSIZE + 2; i++) {
			delete [] board[i];
		}
		delete []board;
	}

	int * const operator[](const int i) {
		return board[i];
	}
	bool operator==(const Board &other) {
		for (int i = 1; i < BSIZE + 1; i++) {
			if (memcmp(&board[i], &other.board[i], sizeof(int) * (BSIZE + 2)) != 0) {
				return false;
			}
		}
		return true;
	}

	void print_board();
	std::vector<Pair*> get_next_moves(COLOR color);
	int update_board(Pair* pos, COLOR color);
	COLOR find_winner();

};

struct BoardHasher
{
	std::size_t operator()(Board*k) const
	{
		size_t hash = 0;
		for (int i = 1; i < BSIZE + 1; i++) {
			for (int j = 1; j < BSIZE + 1; j++) {
				hash = hash * 31 + 2 * (*k)[i][j] + 3;
			}
		}
		return hash;
	}
};

#endif
