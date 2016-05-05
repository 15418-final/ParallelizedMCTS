
#ifndef CUDAGO_H
#define CUDAGO_H

#include "point.h"
#include "deque.h"
#include <vector>
#include <string.h>

enum COLOR {WHITE = 1, BLACK = 2, EMPTY = 0, OUT = 3};

class CudaBoard {
private:
	int dir[4][2] = {{1, 0}, {0, 1}, { -1, 0}, {0, -1}};
	int *board;  	// 1-d array to represent 2d board
	bool *visited;  // same as board
	__device__ __host__ bool canEat(int i, int j, COLOR color, Point* point);
	__device__ __host__ bool isSuicide(int i, int j, COLOR color, Point* point);
	int BSIZE;
	COLOR player;	// current player

	Deque<Point>* q1;
	Deque<Point>* q2;
public:
	__device__ __host__ CudaBoard(int size) {
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

		q1 = new Deque<Point>();
		q2 = new Deque<Point>();
	}

	//copy constructor
	__device__ __host__ CudaBoard(CudaBoard& b) {
		BSIZE = b.get_size();

		int total = (BSIZE + 2) * (BSIZE + 2);
		board = new int[total];
		visited = new bool[total];

		for (int i = 0; i < BSIZE + 2; i++) {
			for (int j = 0; j < BSIZE + 2; j++) {
				board[i * (BSIZE + 2) + j] = b.getBoard(i, j);
				visited[i * (BSIZE + 2) + j] = false;
			}
		}

		player = b.ToPlay();

		q1 = new Deque<Point>();
		q2 = new Deque<Point>();
	}

	__device__ __host__ ~CudaBoard() {
		delete []board;
		delete []visited;
		delete q1;
		delete q2;
	}

	void print_board();
	__device__  Deque<Point>* get_next_moves_device(Point* point);
	std::vector<Point> get_next_moves_host(Point* point);
	__device__ __host__ int update_board(Point pos, Point* point);
	__device__  int score();
	__device__ __host__ COLOR ToPlay() {
		return player;
	}

	__device__ __host__  int get_size() {
		return BSIZE;
	}

	__device__ __host__ int getBoard(int i, int j) {
		return board[i * (BSIZE + 2) + j];
	}

	__device__ __host__ void setBoard(int i, int j, COLOR c) {
		board[i * (BSIZE + 2) + j] = c;
	}

	__device__ __host__ bool isVisited(int i, int j) {
		return visited[i * (BSIZE + 2) + j];
	}

	__device__ __host__ void setVisited(int i, int j, bool b) {
		visited[i * (BSIZE + 2) + j] = b;
	}

	__device__ __host__ void clearVisited() {
		memset(visited, 0, sizeof(bool) * (BSIZE + 2) * (BSIZE + 2));
	}

	__device__ __host__ Point getPoint(Point* point, int i, int j) {
		return *(point + i*(BSIZE+2) + j);
	}
};

#endif
