#include <iostream>
#include <cstdlib>
#include <vector>
#include <stack>
#include <cmath>
#include <algorithm>
#include "mcts.h"

//Exploration parameter
const double C = 1.4;
const double EPSILON = 10e-6;
const int MAX_TRIAL = 1;

TreeNode* Mcts::selection(TreeNode* node) {
	double maxv = -1;
	TreeNode* maxn = NULL;
	int n = node->parent->sims;
	for (TreeNode* c : node->get_children()) {
		double v = (double)c->wins / (c->sims + EPSILON) + C * sqrt(log(n + EPSILON) / (c->sims + EPSILON));
		if (v > maxv) {
			maxv = v;
			maxn = c;
		}
	}
	return maxn;
}
// Typical Monte Carlo Simulation
void Mcts::run_simulation() {
	
}

void Mcts::back_propagation(){
	
}

void Mcts::expand() {
	
}

void Mcts::run_iteration() {
	
}