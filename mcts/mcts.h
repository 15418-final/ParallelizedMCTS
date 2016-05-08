#ifndef MCTS_H
#define MCTS_H

#include <iostream>
#include <cstdlib>
#include <vector>
#include <stack>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#include "point.h"
#include "CudaGo.h"

enum MODE {CPU= 1,  GPU = 2};

typedef struct _threadarg
{
	Point* seq;
	int len;
	int bd_size;
	double win;
	double sim;
	double time;
	int tid;
} thread_arg;

class TreeNode {
private:
	std::vector<Point> sequence;
	std::vector<TreeNode*> children;
	bool expandable;     // unexpaned

public:
	double wins; // Number of wins so far
	double sims; // Number of simulations so far
	TreeNode* parent;
	TreeNode(std::vector<Point> parent_sequence, Point move)
			:  expandable(true), wins(0.0), sims(0.0), parent(NULL) {
		sequence = parent_sequence;
		sequence.push_back(move);		
	}

	TreeNode(std::vector<Point> parent_sequence)
			:  expandable(true), wins(0.0), sims(0.0), parent(NULL) {
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
	struct timespec start, end;
	double maxTime;
	bool abort; 
	int bd_size;
	MODE mode;

public:
	Mcts(MODE m, int size, double time) {
		bd_size = size;
		std::vector<Point> seq;
		root = new TreeNode(seq);
		maxTime = time;
		mode = m;
	}

	Mcts(MODE m, int size, double time, std::vector<Point> seq) {
		bd_size = size;
		root = new TreeNode(seq);
		maxTime = time;
		mode = m;
	}

	~Mcts() {
		delete root;
	}

	//Run MCTS and get 
	Point run();
	
	void run_iteration_gpu(TreeNode* node);
	void run_iteration_cpu(TreeNode* node);
	TreeNode *selection(TreeNode* node);
	void expand(TreeNode* node);
	void back_propagation(TreeNode* node, int win_increase, int sim_increase);

	CudaBoard* get_board(std::vector<Point> sequence, int bd_size);
	bool checkAbort();
	void update(TreeNode* node, double* win, double* sim,  int incre, int thread_num);
};

#endif
