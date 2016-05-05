#include <iostream>
#include "CudaGo.h"
#include "mcts.h"
#include "point.h"

int main(){
	Mcts* m = new Mcts(9);
	m->run();
}
