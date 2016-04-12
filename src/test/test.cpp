#include <iostream>
#include "go.h"
#include "mcts.h"
int main(){
	Board* new_game = new Board();
	new_game->print_board();
	COLOR cur_player = WHITE;
	std::vector<Pair*> moves = new_game->get_next_moves(cur_player);
	if(moves.size() == 0){
		std::cout<<"no moves available" <<std::endl;
	}
	for(Pair* p : moves){
		std::cout<<"("<<p->i<<","<<p->j<<"),";
	}
	Mcts* mcts_manager = new Mcts();
	std::cout<<"Welcome to GO"<<std::endl;
}