
#ifndef CUDAGO_H
#define CUDAGO_H

#include <thrust/device_vector.h>
#include "point.h"

enum COLOR {WHITE = 1, BLACK = 2, EMPTY = 0, OUT = 3};

class CudaBoard {
private:
	int **board;
	bool **visited;
	__device__ __host__ bool canEat(int i, int j, COLOR color);
	__device__ __host__ bool isSuicide(int i, int j, COLOR color);
	int BSIZE;
	COLOR currentPlayer;
public:
	CudaBoard(int size, bool host) {
		BSIZE = size;

		board = new int*[BSIZE + 2];
		visited = new bool*[BSIZE + 2];
		for (int i = 0; i < BSIZE + 2; i++) {
			board[i] = new int[BSIZE + 2];
			visited[i] = new bool[BSIZE + 2];
			if (host) {
				memset(board[i], 0, sizeof(int) * (BSIZE + 2));
				memset(visited[i], 0, sizeof(int) * (BSIZE + 2));
			} else {
				cudaMemset(board[i], 0, sizeof(int) * (BSIZE + 2));
				cudaMemset(visited[i], 0, sizeof(int) * (BSIZE + 2));
			}
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
	bool operator==(const CudaBoard &other) {
		for (int i = 1; i < BSIZE + 1; i++) {
			if (memcmp(&board[i], &other.board[i], sizeof(int) * (BSIZE + 2)) != 0) {
				return false;
			}
		}
		return true;
	}

	void print_board();
	__device__  thrust::device_vector<Point*> get_next_moves_device();
	std::vector<Point*> get_next_moves_host();
	__device__  int update_board(Point* pos);
	__device__  int score();
	__device__  bool EndOfGame();

	__device__  COLOR ToPlay() {
		return currentPlayer;
	}

	__device__  int get_size() {
		return BSIZE;
	}

	__device__ __host__ int get(int i, int j) {
		return board[i][j];
	}

	__device__ __host__ void clearVisited() {
		for (int i = 0; i < BSIZE + 2; i++) {
			for (int j = 0; j < BSIZE + 2; j++) {
				visited[i][j] = false;
			}
		}
	}
};

#endif
