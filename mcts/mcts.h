#ifndef MCTS_H
#define MCTS_H

#include <iostream>
#include <cstdlib>
#include <vector>
#include <stack>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <ctime>

#include "point.h"
#include "CudaGo.h"


class TreeNode {
private:
	std::vector<Point> sequence;
	std::vector<TreeNode*> children;
	bool expandable;     // unexpaned

public:
	int wins; // Number of wins so far
	int sims; // Number of simulations so far
	TreeNode* parent;
	TreeNode(std::vector<Point> parent_sequence, Point move)
			:  expandable(true), wins(0), sims(0), parent(NULL) {
		sequence = parent_sequence;
		sequence.push_back(move);		
	}

	TreeNode(std::vector<Point> parent_sequence)
			:  expandable(true), wins(0), sims(0), parent(NULL) {
		sequence = parent_sequence;
	}

	~TreeNode() {
		for (std::vector<TreeNode*>::iterator it = children.begin(); it != children.end(); it++) {
			delete *it;
		}
		std::vector<Point>().swap(sequence);
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

	std::vector<Point> get_sequence(){
		return sequence;
	}
};


class Mcts {
private:
	TreeNode* root;
	clock_t startTime;
	double maxTime;
	bool abort;

	// board related 
	int bd_size;

	//std::unordered_map<Board*, TreeNode*, BoardHasher> dict;
public:
	Mcts(int size) {
		bd_size = size;
		std::vector<Point> seq;
		root = new TreeNode(seq);
		startTime = clock();
		maxTime = 20000; //milliseconds
	}

	~Mcts() {
		delete root;
	}

	//Run MCTS and get 
	Point run();
	
	//Doing selection using UCT(Upper Confidence bound applied to Trees)
	void run_iteration(TreeNode* node);

	TreeNode *selection(TreeNode* node);
	void expand(TreeNode* node);
	void back_propagation(TreeNode* node, int win_increase, int sim_increase);

	CudaBoard* get_board(std::vector<Point> sequence, int bd_size);
	bool checkAbort();
	std::vector<Point*> generateAllMoves(CudaBoard* cur_board);
	void deleteAllMoves(std::vector<Point*> moves);
	// __device__ GoBoard* get_board(std::vector<SgPoint> sequence);
};

#endif
