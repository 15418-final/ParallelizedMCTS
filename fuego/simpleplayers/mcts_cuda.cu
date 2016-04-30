#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cstdio>
// #include <ctime>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include "mcts.h"
#include "CudaGo.h"
#include "deque.h"

template <typename T>
struct KernelArray
{
	T* _array;
	int _size;
};

//Exploration parameter
double C = 1.4;
double EPSILON = 10e-6;
__constant__ int MAX_TRIAL = 500;
__constant__ int THREAD_NUM = 32;

int MAX_TRIAL_H = 500;


__device__ bool checkAbort();
__device__ Deque<Point*>* generateAllMoves(CudaBoard* cur_board);
__device__ void deleteAllMoves(Deque<Point*>* moves);
__global__ void run_simulation(KernelArray<Point>, int* win_increase, int bd_size);
__device__ CudaBoard* get_board(KernelArray<Point> seq, int bd_size);

template <typename T>
KernelArray<T> convertToKernel(thrust::device_vector<T>& dVec);

SgPoint Mcts::run() {
	// mcts_timer.Start();
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
	return best->get_sequence().back().ToSgPoint();
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
__global__ void run_simulation(KernelArray<Point> seq, int* win_increase, int bd_size) {
	CudaBoard* board = get_board(seq, bd_size);
	COLOR cur_player = board->ToPlay();

	
	int wins = 0;
	for (int i = 0; i < MAX_TRIAL; i++) {
		// bool timeout = false;
		CudaBoard* cur_board = new CudaBoard(*board);
		clock_t start = clock();
		while (true) {
			Deque<Point*>* moves_vec = generateAllMoves(cur_board);
			if (cur_board->EndOfGame() || moves_vec->size() == 0) {
				break;
			}
			//why nxt_move length can be zero? what does endofgame do above?
			// std::cout << "moves_vec length:" << moves_vec->Length() << std::endl;
			Point* nxt_move = (*moves_vec)[moves_vec->begin()];
			
			cur_board->update_board(nxt_move);
			deleteAllMoves(moves_vec);
			// if (checkAbort()) {
			// 	timeout = true;
			// 	break;
			// }
		}
		if (true) {
			int score = cur_board->score(); // Komi set to 0
			if ((score > 0 && cur_player == BLACK)
			        || (score < 0 && cur_player == WHITE)) {
				wins++;
			}
			// totalSimu ++;
		}
		delete cur_board;
	}

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
	CudaBoard* cur_board = get_board(node->get_sequence(),bd_size);

	std::vector<Point*> moves_vec = generateAllMoves(cur_board);
	while (moves_vec.size() > 0) {
		Point* nxt_move = moves_vec.front();
		node->add_children(new TreeNode(node->get_sequence(), *nxt_move));
	}

	deleteAllMoves(moves_vec);
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
				// TreeNode* cudaDeviceNode = NULL;
				// int* cuda_win_increase = NULL;
				// // Use cuda to parallelize
				// cudaMalloc((void **)&cudaDeviceNode, sizeof(*children[i]));
				// cudaMalloc((void **)&cuda_win_increase, sizeof(int));
				// cudaMemcpy(cudaDeviceNode, children[i], sizeof(*children[i]), cudaMemcpyHostToDevice);

				int* cuda_win_increase = NULL;
				cudaMalloc((void **)&cuda_win_increase, sizeof(int));

				thrust::device_vector<Point> dec_seq(children[i]->get_sequence());
				
				run_simulation<<<1,1>>>(convertToKernel(dec_seq), cuda_win_increase, bd_size);
				//cudaFree(cudaDeviceNode);

				int* win_increase = new int[1];
				cudaMemcpy(win_increase, cuda_win_increase, sizeof(int), cudaMemcpyDeviceToHost);

				children[i]->wins += *win_increase;
				children[i]->sims += MAX_TRIAL_H;
				back_propagation(children[i], *win_increase, MAX_TRIAL_H);
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
		// abort = mcts_timer.GetTime() > maxTime;
	}
	return abort;
}

__device__ Deque<Point*>* generateAllMoves(CudaBoard* cur_board) {
	Deque<Point*>* moves_vec = cur_board->get_next_moves_device();
	int len = moves_vec->size();

	// TODO : rand in cuda
	// if (len != 0) {
	// 	srand (time(NULL));
	// 	int swapIndex = rand() % len;
	// 	Point* temp = (*moves_vec)[moves_vec->begin()];
	// 	(*moves_vec)[moves_vec->begin()] = (*moves_vec)[moves_vec->begin() + swapIndex];
	// 	(*moves_vec)[moves_vec->begin() + swapIndex] = temp;
	// }

	return moves_vec;
}

std::vector<Point*> Mcts::generateAllMoves(CudaBoard* cur_board) {
	std::vector<Point*> moves_vec = cur_board->get_next_moves_host();
	int len = moves_vec.size();

	// if (len != 0) {
	// 	srand (time(NULL));
	// 	int swapIndex = rand() % len;
	// 	Point* temp = moves_vec[0];
	// 	moves_vec[0] = moves_vec[swapIndex];
	// 	moves_vec[swapIndex] = temp;
	// }
	return moves_vec;
}

CudaBoard* Mcts::get_board(std::vector<Point> sequence, int bd_size) {
	CudaBoard* bd = new CudaBoard(bd_size);
	for (std::vector<Point>::iterator it = sequence.begin(); it != sequence.end(); it++) {
		bd->update_board(&(*it));
	}
	return bd;
}

__device__ CudaBoard* get_board(KernelArray<Point> sequence, int bd_size) {
	CudaBoard *bd = new CudaBoard(bd_size);
	for (int i = 0; i < sequence._size; i++) {
		bd->update_board(&sequence._array[i]);
	}

	return bd;
}

__device__ void deleteAllMoves(Deque<Point*>* moves) {
	for (int i = moves->begin(); i <= moves->end(); i++) {
		Point* p = (*moves)[i];
		delete p;
	} 
}

void Mcts::deleteAllMoves(std::vector<Point*> moves) {
	for (std::vector<Point*>::iterator it = moves.begin(); it != moves.end(); it++) {
		delete *it;
	}
}

template <typename T>
KernelArray<T> convertToKernel(thrust::device_vector<T>& dVec)
{
    KernelArray<T> kArray;
    kArray._array = thrust::raw_pointer_cast(&dVec[0]);
    kArray._size  = (int) dVec.size();
 
    return kArray;
}

