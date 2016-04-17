#ifndef MCTS_H
#define MCTS_H

#include <iostream>
#include <cstdlib>
#include <vector>
#include <stack>
#include <cmath>
#include <algorithm>
#include <stdlib.h>

#include "SgSystem.h"
#include "SgTimer.h"
#include "SpUtil.h"
#include "SgBlackWhite.h"
#include "GoBoard.h"
#include "GoBoardUtil.h"


class TreeNode {
private:
	std::vector<SgPoint> sequence;
	std::vector<TreeNode*> children;
	bool expandable;     // unexpaned

public:
	int wins; // Number of wins so far
	int sims; // Number of simulations so far
	TreeNode* parent;
	TreeNode(std::vector<SgPoint> parent_sequence, SgPoint move)
			:  expandable(true), wins(0), sims(0), parent(NULL) {
		sequence = parent_sequence;
		if (move != SG_NULLMOVE) {
			sequence.push_back(move);
		}
		
	}

	~TreeNode() {
		for (std::vector<TreeNode*>::iterator it = children.begin(); it != children.end(); it++) {
			delete *it;
		}
		std::vector<SgPoint>().swap(sequence);
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

	std::vector<SgPoint> get_sequence(){
		return sequence;
	}
};


class Mcts {
private:
	TreeNode* root;
	double maxTime;
	SgTimer mcts_timer;
	bool abort;

	// board related 
	SgGrid bd_size;

	//std::unordered_map<Board*, TreeNode*, BoardHasher> dict;
public:
	Mcts(GoBoard& bd, double maxTime) {
		//std::cout<<"Mcts constructor, copied GoBoard"<<std::endl;
		root = new TreeNode(GoBoardUtil::GetSequence(bd), SG_NULLMOVE);
		this->maxTime = maxTime;
		abort = false;

		bd_size = bd.m_size;
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
	int run_simulation(TreeNode* node);

	bool checkAbort();
	SgVector<SgPoint>* generateAllMoves(GoBoard& cur_board);
	GoBoard* get_board(std::vector<SgPoint> sequence);
};

#endif
