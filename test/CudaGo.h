
#ifndef CUDAGO_H
#define CUDAGO_H

#include "point.h"
#include "deque.h"
#include <vector>

enum COLOR {WHITE = 1, BLACK = 2, EMPTY = 0, OUT = 3};

class CudaBoard {
private:
	int *board;  	// 1-d array to represent 2d board
	bool *visited;  // same as board
	bool canEat(int i, int j, COLOR color);
	bool isSuicide(int i, int j, COLOR color);
	int BSIZE;
	COLOR player;
public:
	CudaBoard(int size) {
		BSIZE = size;

		int total = (BSIZE + 2) * (BSIZE + 2);
		board = new int[total];
		visited = new bool[total];

		memset(board, 0, sizeof(int) * total);
		memset(visited, 0, sizeof(bool) * total);

		//set the border
		for (int i = 0; i < BSIZE + 2; i++) {
			board[i * (BSIZE + 2)] = 3;
			board[i * (BSIZE + 2) + BSIZE + 1] = 3;
			board[i] = 3;
			board[(BSIZE + 2) * (BSIZE + 1) + i] = 3;
		}

		player = BLACK; // black play first
	}

	//copy constructor
	CudaBoard(CudaBoard& b) {
		BSIZE = b.get_size();
		
		int total = (BSIZE + 2) * (BSIZE + 2);
		board = new int[total];
		visited = new bool[total];

		for (int i = 0; i < BSIZE + 2; i++) {
			for (int j = 0; j < BSIZE + 2; j++) {
				board[i*(BSIZE+2) + j] = b.getBoard(i, j);
				visited[i*(BSIZE+2) + j] = false;
			}
		}

		player = b.ToPlay();
	}

	~CudaBoard() {
		delete []board;
		delete []visited;
	}

	void print_board();
	 Deque<Point*>* get_next_moves();
	 int update_board(Point* pos);
	 int score();
	 bool EndOfGame();
	void cleanQueue(Deque<Point*>* queue);

	COLOR ToPlay() {
		return player;
	}

	 int get_size() {
		return BSIZE;
	}

	int getBoard(int i, int j) {
		return board[i * (BSIZE + 2) + j];
	}

	void setBoard(int i, int j, COLOR c) {
		board[i * (BSIZE + 2) + j] = c;
	}

	bool isVisited(int i, int j) {
		return visited[i * (BSIZE + 2) + j];
	}

	void setVisited(int i, int j, bool b) {
		visited[i * (BSIZE + 2) + j] = b;
	}

	void clearVisited() {
		memset(visited, 0, sizeof(bool) * (BSIZE + 2) * (BSIZE + 2));
	}
};

#endif
