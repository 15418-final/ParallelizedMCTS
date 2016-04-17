#ifndef MCTS_H
#define MCTS_H

#include <iostream>
#include <cstdlib>
#include <vector>
#include <stack>
#include <cmath>
#include <algorithm>

#include "SgTimer.h"
#include "GoBoard.h"

//#include <omp.h>


class TreeNode {
private:
	GoBoard* board;
	std::vector<TreeNode*> children;
	bool expandable;   // un expaned
public:
	int wins; // Number of wins so far
	int sims; // Number of simulations so far
	TreeNode* parent;
	TreeNode(GoBoard& b):  expandable(true), wins(0), sims(0), parent(NULL) {
		board = new GoBoard(b);		
	}
	
	~TreeNode() {
		for (std::vector<TreeNode*>::iterator it = children.begin(); it != children.end(); it++) {
			delete *it;
		}
		delete board;
		parent = NULL;
	}
	bool is_expandable() {
		return expandable;
	}
	void set_expandable(bool b) {
		expandable = b;
	}
	void add_children(TreeNode* child){
		children.push_back(child);
		child->parent = this;
	}
	std::vector<TreeNode*> get_children() {
		return children;
	}
	GoBoard* get_board(){
		return board;
	}
};


class Mcts {
private:
	TreeNode* root;
	double maxTime;
	SgTimer mcts_timer;
	bool abort;

	//std::unordered_map<Board*, TreeNode*, BoardHasher> dict;
public:
	Mcts(GoBoard& bd, double maxTime) {
		std::cout<<"Mcts constructor, copied GoBoard"<<std::endl;
		
		root = new TreeNode(bd);
		this->maxTime = maxTime;
	}

	~Mcts() {
		delete root;
	}

	//Run MCTS and get 
	SgPoint run();

	//Doing selection using UCT(Upper Confidence bound applied to Trees)
	void run_iteration(TreeNode* node);

	TreeNode *selection(TreeNode* node);
	void expand(TreeNode* node);
	void back_propagation(TreeNode* node, int win_increase, int sim_increase);
	int run_simulation(GoBoard node);

	bool checkAbort();
	SgVector<SgPoint>* generateAllMoves(GoBoard& cur_board);
};

#endif
