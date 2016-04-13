
#include "SgSystem.h"
#include "SpUtil.h"
#include "GoBoardUtil.h"
#include "mcts.h"


//Exploration parameter
const double C = 1.4;
const double EPSILON = 10e-6;
const int MAX_TRIAL = 1;


SgPoint Mcts::run(){
	mcts_timer.Start();
	while(mcts_timer.GetTime() < maxTime){
		run_iteration(root);
	}
	double maxv = 0;
	TreeNode* best = NULL;
	
	for (TreeNode* c : root->get_children()) {
		double v = (double)c->wins / (c->sims + EPSILON);
		if (v > maxv) {
			maxv = v;
			best = c; 
		}
	}
	if(best == NULL){
		return SG_NULLMOVE;
	}
	return best->get_board().GetLastMove();
}

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
void Mcts::run_simulation(TreeNode* node) {
	GoBoard cur_board = node->get_board();//Make a copy of GoBoard
	SgBlackWhite cur_player = cur_board.ToPlay();

	// TODO: parallel this part (leaf parallelization)
	//#pragma omp parallel for private(cur_board)
	for (int i = 0; i < MAX_TRIAL; i++) {
		while (true) {
			if (GoBoardUtil::EndOfGame(cur_board))break;

			SgPointSet moves = SpUtil::GetRelevantMoves(cur_board, cur_board.ToPlay(), true); //UseFilter() set to true
			SgVector<SgPoint>* moves_vec = new SgVector<SgPoint>(); //Init or not?
			moves.ToVector(moves_vec);

			SgPoint nxt_move = (*moves_vec)[rand() % moves_vec->Length()];
			cur_board.Play(nxt_move);
			delete moves_vec;
		}
		float score = GoBoardUtil::Score(cur_board, 0); // Komi set to 0
		if ((score > 0 && cur_player == SG_BLACK)
		        || (score < 0 && cur_player == SG_WHITE)) {
			node->wins++;
		}
		node->sims++;

		//
	}
}

void Mcts::back_propagation(TreeNode* node) {
	int sim_increase = node->sims;
	int win_increase = node->wins;
	bool lv = false;
	while (node->parent != NULL) {
		node = node->parent;
		node->sims += sim_increase;
		if (lv)node->wins += win_increase;
		lv = !lv;
	}
}

void Mcts::expand(TreeNode* node) {
	GoBoard& cur_board = node->get_board();
	SgVector<SgPoint>* moves_vec = new SgVector<SgPoint>(); //Init or not?
	SpUtil::GetRelevantMoves(cur_board, cur_board.ToPlay(), true).ToVector(moves_vec);

	while (moves_vec->Length() > 0) {
		//Copy board
		GoBoard newBoard = cur_board;

		SgPoint nxt_move = moves_vec->PopFront();
		newBoard.Play(nxt_move);
		node->add_children(new TreeNode(newBoard));
	}
	delete moves_vec;
}

void Mcts::run_iteration(TreeNode* node) {
	std::stack<TreeNode*> S;
	S.push(node);
	while (!S.empty()) {
		TreeNode* f = S.top();
		S.pop();
		if (!f->is_expandable()) {
			S.push(selection(node));
		} else {
			// expand current node, run expansion and simulation
			f->set_expandable(false);
			expand(node);

			std::vector<TreeNode*> children = node->get_children();
			for (size_t i = 0; i < children.size(); i++) {
				run_simulation(children[i]);
				back_propagation(children[i]);
			}
		}
	}
}