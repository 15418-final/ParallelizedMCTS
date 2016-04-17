
#include "SgSystem.h"
#include "SpUtil.h"
#include "GoBoardUtil.h"
#include "mcts.h"
#include <stdlib.h>


//Exploration parameter
const double C = 1.4;
const double EPSILON = 10e-6;
const int MAX_TRIAL = 100;


SgPoint Mcts::run() {
	std::cout<<"maxTime:"<<maxTime<<std::endl;
	mcts_timer.Start();
	while (true) {
		// std::cout << "iter" << std::endl;
		run_iteration(root);
		if (checkAbort()) break;
	}
	double maxv = 0;
	TreeNode* best = NULL;
	std::vector<TreeNode*> children = root->get_children();
	for (std::vector<TreeNode*>::iterator it = children.begin(); it != children.end(); it++) {
		TreeNode* c = *it;
		double v = (double)c->wins / (c->sims + EPSILON);
		if (v > maxv) {
			maxv = v;
			best = c;
		}
	}
	if (best == NULL) {
		return SG_NULLMOVE;
	}
	return best->get_board()->GetLastMove();
}

TreeNode* Mcts::selection(TreeNode* node) {
	std::cout << "selection begin" << std::endl;
	double maxv = -10000000;
	TreeNode* maxn = NULL;
	int n = node->sims;
	std::vector<TreeNode*> children = node->get_children();
	for (std::vector<TreeNode*>::iterator it = children.begin(); it != children.end(); it++) {
		TreeNode* c = *it;
		// std::cout<<"c-wins:"<<(double)c->wins<<std::endl;
		// std::cout<<"c-sims:"<<(double)c->sims<<std::endl;
		// std::cout<<"n:"<<(double)n<<std::endl;
		double v = (double)c->wins / (c->sims + EPSILON) + C * sqrt(log(n + EPSILON) / (c->sims + EPSILON));
		// std::cout<<"uct value calculated:"<<v<<std::endl;
		if (v > maxv) {
			maxv = v;
			maxn = c;
		}
	}
	std::cout << "selection end" << std::endl;
	return maxn;
}
// Typical Monte Carlo Simulation
int Mcts::run_simulation(GoBoard cur_board) {//Pass by value. Make a copy of GoBoard
	// std::cout<<"run simulation begin"<<std::endl;
	// GoBoard* cur_board = node->get_board();
	SgBlackWhite cur_player = cur_board.ToPlay();
	int wins = 0;

	// TODO: parallel this part (leaf parallelization)
	//#pragma omp parallel for private(cur_board)
	for (int i = 0; i < MAX_TRIAL; i++) {
		while (true) {
			// SgPointSet moves = SpUtil::GetRelevantMoves(cur_board, cur_board.ToPlay(), true); //UseFilter() set to true
			// // std::cout<<"releveant move:" << moves.Size() <<std::endl;
			// SgVector<SgPoint>* moves_vec = new SgVector<SgPoint>();
			// moves.ToVector(moves_vec);
			SgVector<SgPoint>* moves_vec = generateAllMoves(cur_board);
			if (GoBoardUtil::EndOfGame(cur_board) || moves_vec->Length() == 0){
				// std::cout<<"simulation reach end"<<std::endl;
				break;
			}

			//why nxt_move length can be zero? what does endofgame do above?
			// std::cout<<"moves_vec length:"<<moves_vec->Length()<<std::endl;
			SgPoint nxt_move = (*moves_vec)[rand() % moves_vec->Length()];
			// std::cout << "In simu:nxt_move get:" << nxt_move << std::endl;
			// std::cout << "In simu:nxt color:" << cur_board.ToPlay() << std::endl;
			// std::cout<<"before play"<<std::endl;
			 // std::cout<<nxt_move<<std::endl;
			cur_board.Play(nxt_move);
			 //std::cout<<"after play"<<std::endl;
			if (checkAbort()) break;
			delete moves_vec;
		}
		float score = GoBoardUtil::Score(cur_board, 0); // Komi set to 0
		// std::cout<<"score calculated:"<<score<<std::endl;
		if ((score > 0 && cur_player == SG_BLACK)
		        || (score < 0 && cur_player == SG_WHITE)) {
			wins++;
		}
	}
	return wins;
	// std::cout << "run_simulation end" << std::endl;
}

void Mcts::back_propagation(TreeNode* node, int win_increase, int sim_increase) {
	bool lv = false;
	while (node->parent != NULL) {
		node = node->parent;
		node->sims += sim_increase;
		if (lv)node->wins += win_increase;
		lv = !lv;
	}
}

void Mcts::expand(TreeNode* node) {
	//std::cout << "expand begin" << std::endl;
	GoBoard* cur_board = node->get_board();
	//std::cout<<"cur_board:"<<cur_board<<std::endl;
	SgVector<SgPoint>* moves_vec = generateAllMoves(*cur_board);
	//SpUtil::GetRelevantMoves(*cur_board, cur_board->ToPlay(), true).ToVector(moves_vec);
	//std::cout<<"expand: nxt moves num:"<<moves_vec->Length()<<std::endl;
	while (moves_vec->Length() > 0) {
		//Copy board
		GoBoard* newBoard = new GoBoard(*cur_board);
		SgPoint nxt_move = moves_vec->PopFront();
		// std::cout << "In expand:nxt_move get:" << nxt_move << std::endl;
		//std::cout << "In expand:nxt color:" << newBoard->ToPlay() << std::endl;
		newBoard->Play(nxt_move);
		//std::cout << "after play" << std::endl;
		node->add_children(new TreeNode(*newBoard));
		//std::cout << "after add children" << std::endl;
		delete newBoard;
	}
	delete moves_vec;

	//std::cout << "expand end with children num:" << node->get_children().size() << std::endl;
}

void Mcts::run_iteration(TreeNode* node) {
	std::stack<TreeNode*> S;
	S.push(node);

	while (!S.empty()) {
		TreeNode* f = S.top();
		S.pop();
		if (!f->is_expandable()) {
		//	std::cout<<"select f:"<<f<<std::endl;
			S.push(selection(f));
		} else {
			// expand current node, run expansion and simulation
			f->set_expandable(false);
			expand(f);
			//std::cout<<"expand f end:"<<f<<std::endl;

			std::vector<TreeNode*> children = f->get_children();
			for (size_t i = 0; i < children.size(); i++) {
				int win_increase = run_simulation(*(children[i]->get_board()));
				children[i]->wins += win_increase;
				children[i]->sims += MAX_TRIAL;
				back_propagation(children[i], win_increase, MAX_TRIAL);
			}
		}

		if (checkAbort()) break;
	}

	std::cout << "run_iteration end"<< std::endl;
}

bool Mcts::checkAbort() {
	if (!abort) {
		abort = mcts_timer.GetTime() > maxTime;
	}
	return abort;
}

SgVector<SgPoint>* Mcts::generateAllMoves(GoBoard& cur_board) {
	SgPointSet moves = SpUtil::GetRelevantMoves(cur_board, cur_board.ToPlay(), true);
	SgVector<SgPoint>* moves_vec = new SgVector<SgPoint>();
    moves.ToVector(moves_vec);
    int len = moves_vec->Length();
    if (len != 0) {
    //	std::cout<<"swap"<<std::endl;
	    srand (time(NULL));
	    int swapIndex = rand() % len;
	    moves_vec->swap(0, swapIndex);
	}
    return moves_vec;
}


