#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cstdio>
#include <ctime>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include "mcts.h"

//Exploration parameter
__constant__ double C = 1.4;
__constant__ double EPSILON = 10e-6;
__constant__ int MAX_TRIAL = 500;
__constant__ int THREAD_NUM = 32;


__device__ bool checkAbort();
__device__ SgVector<SgPoint>* generateAllMoves(GoBoard& cur_board);
__device__ GoBoard* get_board(std::vector<SgPoint> sequence, int bd_size);
__global__ void run_simulation(TreeNode* node, int* win_increase);

SgPoint Mcts::run() {
	mcts_timer.Start();
	while (true) {
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
	// std::cout << "Total simulation runs:" << totalSimu << std::endl;
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
		double v = (double)c->wins / (c->sims + EPSILON) + C * sqrt(log(n + EPSILON) / (c->sims + EPSILON));
		if (v > maxv) {
			maxv = v;
			maxn = c;
		}
	}
	std::cout << "selection end" << std::endl;
	return maxn;
}
// Typical Monte Carlo Simulation
__global__ void run_simulation(Mcts* m, TreeNode* node, int* win_increase) {
	GoBoard* board = get_board(node->get_sequence(), m->getBoardSize());
	SgBlackWhite cur_player = board->ToPlay();
	int wins = 0;
	for (int i = 0; i < MAX_TRIAL; i++) {
		bool timeout = false;
		GoBoard* cur_board = new GoBoard(*board);
		clock_t start = clock();
		while (true) {
			SgVector<SgPoint>* moves_vec = generateAllMoves(*cur_board);
			if (GoBoardUtil::EndOfGame(*cur_board) || moves_vec->Length() == 0) {
				break;
			}
			//why nxt_move length can be zero? what does endofgame do above?
			// std::cout << "moves_vec length:" << moves_vec->Length() << std::endl;
			SgPoint nxt_move = (*moves_vec)[rand() % moves_vec->Length()];
			
			cur_board->Play(nxt_move);
			delete moves_vec;
			// if (checkAbort()) {
			// 	timeout = true;
			// 	break;
			// }
		}
		if (!timeout) {
			float score = GoBoardUtil::Score(*cur_board, 0); // Komi set to 0
			if ((score > 0 && cur_player == SG_BLACK)
			        || (score < 0 && cur_player == SG_WHITE)) {
				wins++;
			}
			// totalSimu ++;
		}
		delete cur_board;
	}
	delete board;
	return;
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
	GoBoard* cur_board = get_board(node->get_sequence(),bd_size);

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
				TreeNode* cudaDeviceNode = NULL;
				int* cuda_win_increase = NULL;
				// Use cuda to parallelize
				cudaMalloc((void **)&cudaDeviceNode, sizeof(*children[i]));
				cudaMalloc((void **)&cuda_win_increase, sizeof(int));
				cudaMemcpy(cudaDeviceNode, children[i], sizeof(*children[i]), cudaMemcpyHostToDevice);

				run_simulation<<<1,1>>>(cudaDeviceNode,cuda_win_increase);
				cudaFree(cudaDeviceNode);

				int* win_increase = new int[1];
				cudaMemcpy(win_increase, cuda_win_increase, sizeof(int), cudaMemcpyDeviceToHost);

				children[i]->wins += *win_increase;
				children[i]->sims += MAX_TRIAL;
				back_propagation(children[i], *win_increase, MAX_TRIAL);
				delete cuda_win_increase;
				delete win_increase;
				if(checkAbort())break;
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

__device__ SgVector<SgPoint>* generateAllMoves(GoBoard& cur_board) {
	//std::cout<<cur_board.m_size<<std::endl;
	SgPointSet moves = SpUtil::GetRelevantMoves(cur_board, cur_board.ToPlay(), true);

	SgVector<SgPoint>* moves_vec = new SgVector<SgPoint>();
	moves.ToVector(moves_vec);
	int len = moves_vec->Length();

	if (len != 0) {
		srand (time(NULL));
		int swapIndex = rand() % len;
		moves_vec->swap(0, swapIndex);
	}
	return moves_vec;
}

__device__ GoBoard* get_board(std::vector<SgPoint> sequence, int bd_size) {
	GoBoard* bd = new GoBoard(bd_size);
	for (std::vector<SgPoint>::iterator it = sequence.begin(); it != sequence.end(); it++) {
		bd->Play(*it);
	}
	return bd;
}


