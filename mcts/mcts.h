#ifndef MCTS_H
#define MCTS_H

class TreeNode {
private:
	std::vector<TreeNode*> children;
	bool expandable;   // un expaned
public:
	int wins; // Number of wins so far
	int sims; // Number of simulations so far
	bool is_leaf;
	TreeNode* parent;
	TreeNode(): expandable(true), wins(0), sims(0), is_leaf(false), parent(NULL) {
	}
	
	~TreeNode() {
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
};


class Mcts {
private:
	TreeNode* root;
	//std::unordered_map<Board*, TreeNode*, BoardHasher> dict;
public:
	Mcts() {
		root = new TreeNode();

		// White first
		//cur_color = WHITE;
	}
	~Mcts() {
		delete(root);
	}

	//Doing selection using UCT(Upper Confidence bound applied to Trees)
	void run_iteration();

	TreeNode *selection();
	void expand();
	void back_propagation();
	void run_simulation();
};

#endif
