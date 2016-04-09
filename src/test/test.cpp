#include <iostream>
#include "go.h"
#include "mcts.h"
int main(){
	Board* new_game = new Board();
	new_game->print_board();
	Mcts* mcts_manager = new Mcts();
	std::cout<<"Welcome to GO"<<std::endl;
}