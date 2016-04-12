#include <iostream>
#include <cstdlib>
#include <vector>
#include <stack>
#include <cmath>
#include <algorithm>
#include "mcts.h"
#include "go.h"
//Exploration parameter
const double C = 1.4;
const double EPSILON = 10e-6;
const int MAX_TRIAL = 1;

TreeNode* Mcts::selection(TreeNode* node) {
	double maxv = -1;
	TreeNode* maxn = NULL;
	for (TreeNode* c : node->get_children()) {
		double v = (double)c->wins / (c->sims + EPSILON) + C * sqrt(log(node->parent->sims + EPSILON) / (c->sims + EPSILON));
		if (v > maxv) {
			maxv = v;
			maxn = c;
		}
	}
	return maxn;
}
// Typical Monte Carlo Simulation
void Mcts::run_simulation(TreeNode* node, COLOR color) {
	Board* cur_state = node->get_state();
	COLOR sim_color = color;
	// TODO: parallel this part
	for (int i = 0; i < MAX_TRIAL; i++) {
		while (true) {
			std::vector<Pair*> moves = cur_state->get_next_moves(color);
			if (moves.size() == 0)break;
			Pair* nxt_move = moves[rand() % moves.size()];
			cur_state->update_board(nxt_move, sim_color);
			sim_color = static_cast<COLOR>(sim_color^3);

			for(Pair* pair : moves){
				delete pair;
			}
		}
		COLOR winner = cur_state->find_winner();
		if (winner == color) {
			node->wins++;
		}
		node->sims++;
	}
}

void Mcts::back_propagation(TreeNode* node){
	int sim_increase = node->sims;
	int win_increase = node->wins;
	bool lv = false;
	while(node->parent != NULL){
		node = node->parent;
		node->sims += sim_increase;
		if(lv)node->wins += win_increase;
		lv = !lv;
	}
}

void Mcts::expand(TreeNode* node, Board* cur_state, COLOR color) {
	std::vector<Pair*> moves = cur_state->get_next_moves(color);
	while (moves.size() > 0) {
		Pair* nxt_move = moves.back();
		cur_state->update_board(nxt_move, color);
		node->add_children(new TreeNode(cur_state));
		moves.pop_back();
	}
}

void Mcts::run_iteration(TreeNode* node, COLOR color) {
	std::stack<TreeNode*> S;
	S.push(node);
	while (!S.empty()) {
		TreeNode* f = S.top();
		S.pop();
		if (!f->is_expandable()) {
			S.push(selection(node));
		} else {
			// expand current node, run expansion and simulation
			f->set_expandable(false);
			Board* cur_state = new Board(*(f->get_state()));
			expand(node, cur_state, color);

			// TODO: parallel simulation of all expanded children
			COLOR cur_color = static_cast<COLOR>(color ^ 3);
			std::vector<TreeNode*> children = node->get_children();
			for (size_t i = 0; i < children.size(); i++) {
				run_simulation(children[i], cur_color);
				back_propagation(children[i]);
			}
			delete cur_state;
		}
		color = static_cast<COLOR>(color^3);
	}
}