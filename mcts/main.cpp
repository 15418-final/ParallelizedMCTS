#include <iostream>
#include "CudaGo.h"
#include "mcts.h"
#include "point.h"

int main(){
	Mcts* m = new Mcts(9, 6*20*1000);
	m->run();
}
