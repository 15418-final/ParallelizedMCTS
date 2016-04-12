#include <iostream>
#include <cstdlib>
#include <vector>
#include <stack>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "mcts.h"
#include "../fuego/simpleplayers/SpUtil.h"

//Exploration parameter
const double C = 1.4;
const double EPSILON = 10e-6;
const int MAX_TRIAL = 1;

TreeNode* Mcts::selection() {
	return NULL;
}
// Typical Monte Carlo Simulation
void Mcts::run_simulation(TreeNode* node) {
	GoBoard& cur_board = node->get_board();
	SgBlackWhite cur_player = cur_board.ToPlay();

	// TODO: parallel this part
	#pragma omp parallel for private(cur_board)
	for (int i = 0; i < MAX_TRIAL; i++) {
		while (true) {
			if (GoBoardUtil::EndOfGame(cur_board))break;

			SgPointSet moves = SpUtil::GetRelevantMoves(cur_board, cur_board.ToPlay(), true);
			SgVector<SgPoint>* moves_vec; //Init or not?
			moves.ToVector(moves_vec);

			SgPoint nxt_move = moves_vec[rand() % moves_vec.size()];
			cur_board.Play(nxt_move);
			delete moves_vec;
		}
		float socre = cur_board.Score(cur_board, 0); // Komi set to 0
		if ((score > 0 && cur_player == SG_BLACK)
		        || (score < 0 && cur_player == SG_WHITE)) {
			node->wins++;
		}
		node->sims++;
	}
}

void Mcts::back_propagation() {

}

void Mcts::expand() {

}

void Mcts::run_iteration() {

}