#include "CudaGo.h"
#include "point.h"
#include "deque.h"
#include <stdlib.h>    
#include <time.h>
#include <ctime>	

int main() {
	srand (time(NULL));

	clock_t  start = clock();
	CudaBoard* board = new CudaBoard(9);
	Deque<Point*>* moves = board->get_next_moves();
	int step = 0;
	while (moves->size() > 0) {
		//Point* p = (*moves)[rand() % moves->size()];
		Point* p = moves->front();
		board->update_board(p);
		board->print_board();
		moves = board->get_next_moves();
		step++;
	}
	clock_t end = clock();
	printf("time: %f ms\n",  (end - start) / (double)(CLOCKS_PER_SEC / 1000));
	printf("End of game.\n");
	if (board->score() > 0) {
		printf("black win\n");
	} else {
		printf("white win\n");
	}
	printf("total step:%d\n", step);
}