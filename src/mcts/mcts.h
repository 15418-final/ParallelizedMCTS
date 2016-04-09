#ifndef MCTS_H
#define MCTS_H
#include "go.h"
#include <unordered_map>

class TreeNode {
private:
	Board* state;
	std::vector<TreeNode*> children;
	bool visited;
public:
	int wins; // Number of wins so far
	int sims; // Number of simulations so far
	bool is_leaf;
	TreeNode* parent;
	TreeNode(): state(NULL), visited(false), wins(0), sims(0), is_leaf(false), parent(NULL) {
	}
	TreeNode(Board *b): state(b), visited(false), wins(0), sims(0), is_leaf(false), parent(NULL) {
	}
	~TreeNode() {
		delete []state;
		for (TreeNode* tn : children) {
			delete tn;
		}
		delete parent;
	}
	bool is_visited() {
		return visited;
	}
	void set_visited() {
		visited = true;
	}
	void add_children(TreeNode* child){
		children.push_back(child);
	}
	std::vector<TreeNode*> get_children() {
		return children;
	}
	Board* get_state() {
		return state;
	}
};


class Mcts {
private:
	TreeNode* root;
	COLOR cur_color;
	std::unordered_map<Board*, TreeNode*, BoardHasher> dict;
public:
	Mcts() {
		root = new TreeNode();

		// White first
		cur_color = WHITE;
	}
	~Mcts() {
		delete(root);
	}

	//Doing selection using UCT(Upper Confidence bound applied to Trees)
	void run_iteration(TreeNode* node, COLOR color);

	TreeNode *selection(TreeNode* node);
	void back_propagation(TreeNode* node);
	void run_simulation(TreeNode* node, COLOR color);
};

#endif
