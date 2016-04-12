#ifndef MCTS_H
#define MCTS_H
#include "go.h"
#include <unordered_map>

class TreeNode {
private:
	Board* state;
	std::vector<TreeNode*> children;
	bool expandable;   // un expaned
public:
	int wins; // Number of wins so far
	int sims; // Number of simulations so far
	bool is_leaf;
	TreeNode* parent;
	TreeNode(): state(NULL), expandable(true), wins(0), sims(0), is_leaf(false), parent(NULL) {
	}
	TreeNode(Board *b): state(b), expandable(true), wins(0), sims(0), is_leaf(false), parent(NULL) {
	}
	~TreeNode() {
		delete []state;
		for (TreeNode* tn : children) {
			delete tn;
		}
		delete parent;
	}
	bool is_expandable() {
		return expandable;
	}
	void set_expandable(bool b) {
		expandable = b;
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
	void expand(TreeNode* node, Board* cur_state, COLOR color);
	void back_propagation(TreeNode* node);
	void run_simulation(TreeNode* node, COLOR color);
};

#endif
