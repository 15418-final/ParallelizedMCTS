#include <iostream>
#include "GoBoard.h"
#include "mcts.h"
#include "point.h"

int main(){
	Mcts* m = new Mcts(11);
	m->run();
}
