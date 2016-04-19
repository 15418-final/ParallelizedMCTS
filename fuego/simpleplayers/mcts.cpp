#include <cstdio>
#include <ctime>
// #include <chrono>
#include "mcts.h"

//Exploration parameter
const double C = 1.4;
const double EPSILON = 10e-6;
const int MAX_TRIAL = 500;

const int THREAD_NUM = 1; //ghc41:12*6

static int totalSimu = 0;

SgPoint Mcts::run() {
	std::cout << "maxTime:" << maxTime << std::endl;
	mcts_timer.Start();
	while (true) {
		std::cout << "iter" << std::endl;
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
	std::cout << "Total simulation runs:" << totalSimu << std::endl;
	return best->get_sequence().back();
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
int Mcts::run_simulation(TreeNode* node) {
	// std::cout << "run simulation in parallel begin" << std::endl;
	GoBoard* board = get_board(node->get_sequence());
	SgBlackWhite cur_player = board->ToPlay();
	int wins = 0;
	#pragma omp parallel for schedule(dynamic, 5) num_threads(THREAD_NUM)
	for (int i = 0; i < MAX_TRIAL; i++) {
		bool timeout = false;
		GoBoard* cur_board = new GoBoard(*board);
		// printf("threadid:%d\n", omp_get_thread_num());
		clock_t start = clock();
		while (true) {
			// SgPointSet moves = SpUtil::GetRelevantMoves(cur_board, cur_board.ToPlay(), true); //UseFilter() set to true
			// // std::cout<<"releveant move:" << moves.Size() <<std::endl;
			// SgVector<SgPoint>* moves_vec = new SgVector<SgPoint>();
			// moves.ToVector(moves_vec);
			SgVector<SgPoint>* moves_vec = generateAllMoves(*cur_board);
			if (GoBoardUtil::EndOfGame(*cur_board) || moves_vec->Length() == 0) {
				// std::cout << "simulation reach end" << std::endl;
				break;
			}

			//why nxt_move length can be zero? what does endofgame do above?
			// std::cout << "moves_vec length:" << moves_vec->Length() << std::endl;
			SgPoint nxt_move = (*moves_vec)[rand() % moves_vec->Length()];
			// std::cout << "In simu:nxt_move get:" << nxt_move << std::endl;
			// std::cout << "In simu:nxt color:" << cur_board->ToPlay() << std::endl;
			// std::cout << "before play" << std::endl;
			// std::cout<<nxt_move<<std::endl;
			cur_board->Play(nxt_move);
			// std::cout << "after play" << std::endl;
			delete moves_vec;
			if (checkAbort()) {
				timeout = true;
				break;
			}
		}
		if (!timeout) {
			// std::cout << "Time :" <<  1000.0 * double(clock() - start) / CLOCKS_PER_SEC << std::endl;
			float score = GoBoardUtil::Score(*cur_board, 0); // Komi set to 0
			// std::cout<<"score calculated:"<<score<<std::endl;
			if ((score > 0 && cur_player == SG_BLACK)
			        || (score < 0 && cur_player == SG_WHITE)) {
				#pragma omp atomic
				wins++;
			}
			// if (checkAbort()) break;
			#pragma omp atomic
			totalSimu ++;
		}
		// std::cout<<"Board size at end of game(in bytes):"<<cur_board->getMySize()<<std::endl;
		delete cur_board;
	}
	std::cout << "run_simulation end with wins:" << wins << std::endl;
	delete board;
	return wins;
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
	std::cout << "expand begin" << std::endl;
	GoBoard* cur_board = get_board(node->get_sequence());

	//std::cout<<"cur_board:"<<cur_board<<std::endl;
	SgVector<SgPoint>* moves_vec = generateAllMoves(*cur_board);
	//SpUtil::GetRelevantMoves(*cur_board, cur_board->ToPlay(), true).ToVector(moves_vec);
	//std::cout<<"expand: nxt moves num:"<<moves_vec->Length()<<std::endl;
	while (moves_vec->Length() > 0) {
		SgPoint nxt_move = moves_vec->PopFront();
		// std::cout << "In expand:nxt_move get:" << nxt_move << std::endl;
		//std::cout << "In expand:nxt color:" << newBoard->ToPlay() << std::endl;
		//std::cout << "after play" << std::endl;
		node->add_children(new TreeNode(node->get_sequence(), nxt_move));
		//std::cout << "after add children" << std::endl;
	}
	delete moves_vec;
	delete cur_board;

	std::cout << "expand end with children num:" << node->get_children().size() << std::endl;
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
				int win_increase = run_simulation(children[i]);
				children[i]->wins += win_increase;
				children[i]->sims += MAX_TRIAL;
				back_propagation(children[i], win_increase, MAX_TRIAL);
			}
		}

		if (checkAbort()) break;
	}

	std::cout << "run_iteration end:" << std::endl;
}

bool Mcts::checkAbort() {
	if (!abort) {
		abort = mcts_timer.GetTime() > maxTime;
	}
	return abort;
}

SgVector<SgPoint>* Mcts::generateAllMoves(GoBoard& cur_board) {
	//std::cout<<cur_board.m_size<<std::endl;
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

GoBoard* Mcts::get_board(std::vector<SgPoint> sequence) {
	GoBoard* bd = new GoBoard(bd_size);
	for (std::vector<SgPoint>::iterator it = sequence.begin(); it != sequence.end(); it++) {
		bd->Play(*it);
	}
	return bd;
}


