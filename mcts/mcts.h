#ifndef MCTS_H
#define MCTS_H

class TreeNode {
private:
	GoBoard& board;
	std::vector<TreeNode*> children;
	bool expandable;   // un expaned
public:
	int wins; // Number of wins so far
	int sims; // Number of simulations so far
	TreeNode* parent;
	TreeNode(GoBoard& b): board(b), expandable(true), wins(0), sims(0), parent(NULL) {
	}
	
	~TreeNode() {
		for (std::vector<TreeNode*>::iterator it = children.begin(); it != children.end(); it++) {
			delete *it;
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
	GoBoard& get_board(){
		return board;
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

	TreeNode* selection();
	void expand();
	void back_propagation();
	void run_simulation(TreeNode* node);
};

#endif
