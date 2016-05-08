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

	while (step < 80) {
		cpu = new Mcts(CPU, 9, 15 * 1000, seq);
		p = cpu->run();
		step++;
		printf("cpu : (%d,%d)\n", p.i, p.j);
		seq.push_back(p);
		gpu = new Mcts(GPU, 9, 15 * 1000, seq);
		p = gpu->run();
		step++;
		seq.push_back(p);
		printf("gpu : (%d,%d)\n", p.i, p.j);
		delete cpu;
		delete gpu;
	}
	printf("score:%d\n", board.score());
}

