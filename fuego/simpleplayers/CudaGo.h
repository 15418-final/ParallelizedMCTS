
#ifndef CUDAGO_H
#define CUDAGO_H

#include "point.h"
#include "deque.h"

enum COLOR {WHITE = 1, BLACK = 2, EMPTY = 0, OUT = 3};

class CudaBoard {
private:
	int **board;
	bool **visited;
	bool canEat(int i, int j, COLOR color);
	bool isSuicide(int i, int j, COLOR color);
	int BSIZE;
	COLOR currentPlayer;
public:
	CudaBoard(int size) {
		BSIZE = size;

		board = new int*[BSIZE + 2];
		visited = new bool*[BSIZE + 2];
		for (int i = 0; i < BSIZE + 2; i++) {
			board[i] = new int[BSIZE + 2];
			visited[i] = new bool[BSIZE + 2];

			memset(board[i], 0, sizeof(int) * (BSIZE + 2));
			memset(visited[i], 0, sizeof(bool) * (BSIZE + 2));
		}

		//set the border
		for (int i = 0; i < BSIZE + 2; i++) {
			board[i][0] = 3;
			board[0][i] = 3;
			board[i][BSIZE + 1] = 3;
			board[BSIZE + 1][i] = 3;
		}

		currentPlayer = BLACK; // black first
	}

	//copy constructor
	CudaBoard(CudaBoard& b) {
		BSIZE = b.get_size();
		board = new int*[BSIZE + 2];
		visited = new bool*[BSIZE + 2];

		for (int i = 0; i < BSIZE + 2; i++) {
			board[i] = new int[BSIZE + 2];
			visited[i] = new bool[BSIZE + 2];

			for (int j = 0; j < BSIZE + 2; j++) {
				board[i][j] = b.get(i, j);
				visited[i][j] = false;
			}
		}

		currentPlayer = b.ToPlay();
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

	void print_board();
	Deque<Point*>* get_next_moves_device();
	std::vector<Point*> get_next_moves_host();
	int update_board(Point* pos);
	int score();
	bool EndOfGame();

	COLOR ToPlay() {
		return currentPlayer;
	}

	int get_size() {
		return BSIZE;
	}

	int get(int i, int j) {
		return board[i][j];
	}

	void clearVisited() {
		for (int i = 0; i < BSIZE + 2; i++) {
			memset(visited[i], 0, sizeof(bool) * (BSIZE + 2));
		}
	}
};

#endif
