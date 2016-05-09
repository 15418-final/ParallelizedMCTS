#include <stdio.h>
#include "CudaGo.h"
#include "mcts.h"
#include "point.h"

int main() {
	Mcts* cpu;
	Mcts* gpu;
	Point p;
	CudaBoard board(9);
	int step = 0;
	std::vector<Point> seq;

	while (step < 120) {
		gpu = new Mcts(GPU, 9, 60 * 1000, seq);
		p = gpu->run();
		step++;
		printf("gpu : (%d,%d)\n", p.i, p.j);
		seq.push_back(p);
		board.update_board(p);
		board.print_board();
		cpu = new Mcts(CPU, 9, 60 * 1000, seq);
		p = cpu->run();
		step++;
		seq.push_back(p);
		printf("cpu : (%d,%d)\n", p.i, p.j);
		board.update_board(p);
		board.print_board();
		delete cpu;
		delete gpu;
	}
	printf("score:%d\n", board.score());
}

