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
	//IN MCTS, max_trail is 1
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

void Mcts::run_iteration(TreeNode* node, COLOR color) {
	std::stack<TreeNode*> S;
	S.push(node);
	while (!S.empty()) {
		TreeNode* f = S.top();
		S.pop();
		if (f->is_visited()) {
			//If node is visited, run selection
			S.push(selection(node));
		} else {
			//If node is not visited yet, run expansion and simulation
			f->set_visited();
			Board* cur_state = new Board(*(f->get_state()));
			std::vector<Pair*> moves = cur_state->get_next_moves(color);
			for (int i = 0; i < std::min<unsigned int>(5, moves.size()); i++) {
				int chosen = rand() % moves.size();
				Pair* nxt_move = moves[chosen];
				moves.erase(moves.begin() + chosen);
				cur_state->update_board(nxt_move, color);
				if (dict.find(cur_state) == dict.end()) {
					dict[cur_state] = new TreeNode(cur_state);
					run_simulation(dict[cur_state],color);
					back_propagation(dict[cur_state]);
				}
				f->add_children(dict[cur_state]);
				delete cur_state;
				cur_state = new Board(*(f->get_state()));
			}
			delete cur_state;
		}
		color = static_cast<COLOR>(color^3);
	}
}