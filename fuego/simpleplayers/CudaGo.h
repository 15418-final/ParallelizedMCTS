
#ifndef CUDAGO_H
#define CUDAGO_H

#include <thrust/device_vector.h>
#include "point.h"

enum COLOR {WHITE = 1, BLACK = 2, EMPTY = 0, OUT = 3};

class CudaBoard {
private:
	int **board;
	bool canEat(int i, int j, COLOR color);
	bool isSuicide(int i, int j, COLOR color);
	int BSIZE;
	COLOR currentPlayer;
public:
	CudaBoard(int size, COLOR player) {
		BSIZE = size;

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

		currentPlayer = player;
	}

	//copy constructor
	CudaBoard(CudaBoard& b) {
		BSIZE = b.get_size();
		board = new int*[BSIZE + 2];
		for (int i = 0; i < BSIZE + 2; i++) {
			board[i] = new int[BSIZE + 2];
			memcpy(&board[i], &b.board[i], sizeof(int) * (BSIZE + 2));
		}

		currentPlayer = b.get_player();
	}

	~CudaBoard() {
		for (int i = 0; i < BSIZE + 2; i++) {
			delete [] board[i];
		}
		delete []board;
	}

	int * const operator[](const int i) {
		return board[i];
	}
	bool operator==(const CudaBoard &other) {
		for (int i = 1; i < BSIZE + 1; i++) {
			if (memcmp(&board[i], &other.board[i], sizeof(int) * (BSIZE + 2)) != 0) {
				return false;
			}
		}
		return true;
	}

	void print_board();
	thrust::device_vector<Point*> get_next_moves(COLOR color);
	int update_board(Point* pos, COLOR color);
	int score();
	bool EndOfGame();

	int get_size() {
		return BSIZE;
	}

	COLOR get_player() {
		return currentPlayer;
	}

};

#endif
