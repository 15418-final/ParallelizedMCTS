#include <stdio.h>
#include <cstdio>
// #include <ctime>
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
int MAX_TRIAL = 5;
int THREAD_NUM = 32;
int MAX_TRIAL_H = 50;


bool checkAbort();
Deque<Point*>* generateAllMoves(CudaBoard* cur_board);
void deleteAllMoves(Deque<Point*>* moves);
void run_simulation(std::vector<Point> v, int* win_increase, int bd_size);
CudaBoard* get_board(std::vector<Point> seq, int bd_size);


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
void run_simulation(std::vector<Point> seq, int* cuda_rand_nums, int* win_increase, int bd_size) {
	CudaBoard* board = get_board(seq, bd_size);
	COLOR cur_player = board->ToPlay();
	// if (threadIdx.x == 0 && blockIdx.x == 0) printf("10th random number in array:%d\n", cuda_rand_nums[10]);
	*win_increase = 0;
	for (int i = 0; i < MAX_TRIAL; i++) {
		// bool timeout = false;
		CudaBoard* cur_board = new CudaBoard(*board);
		clock_t start = clock();
		// printf("Start a simulation\n");
		while (true) {
			Deque<Point*>* moves_vec = generateAllMoves(cur_board);
			// printf("generate moves done:%d\n", moves_vec->size());
			if (cur_board->EndOfGame() || moves_vec->size() == 0) {
				break;
			}
			//why nxt_move length can be zero? what does endofgame do above?
			// std::cout << "moves_vec length:" << moves_vec->Length() << std::endl;
			Point* nxt_move = *(moves_vec->begin());
			// printf("next move get:%d, %d\n", nxt_move->i, nxt_move->j);
			cur_board->update_board(nxt_move);
			// printf("move made\n");
			deleteAllMoves(moves_vec);
			// if (checkAbort()) {
			// 	timeout = true;
			// 	break;
			// }
		}
		if (true) {
			int score = cur_board->score();
			if ((score > 0 && cur_player == BLACK)
			        || (score < 0 && cur_player == WHITE)) {
				*win_increase = *win_increase + 1;
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
	CudaBoard* cur_board = get_board(node->get_sequence(), bd_size);

	std::vector<Point*> moves_vec = generateAllMoves(cur_board);
	std::cout << "moves generated:" << moves_vec.size() << std::endl;
	while (moves_vec.size() > 0) {
		Point* nxt_move = moves_vec.back();
		node->add_children(new TreeNode(node->get_sequence(), *nxt_move));
		moves_vec.pop_back();
	}
	// std::cout << "children add done" << std::endl;
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
			std::cout << "expand f end:" << f << std::endl;

			std::vector<TreeNode*> children = f->get_children();
			for (size_t i = 0; i < children.size(); i++) {
				// std::cout << "Cuda malloc done" << std::endl;

				std::vector<Point> dec_seq(children[i]->get_sequence());

				int* rand_nums = new int[100];
				srand (time(NULL));
				for (int r = 0; r < 100; r++) {
					rand_nums[r] = rand() % (bd_size * bd_size);
				}
				int* cuda_rand_nums;
				cuda_rand_nums = (int*)malloc(sizeof(int) * 100);
				memcpy(cuda_rand_nums, rand_nums, sizeof(int) * 100);
				// std::cout << "rand_nums[10]:" << rand_nums[10] << std::endl;
				int* cuda_win_increase = NULL;
				cuda_win_increase = (int*)malloc(sizeof(int));
				// run_simulation<<<1,1>>>(convertToKernel(dec_seq), cuda_rand_nums, cuda_win_increase, bd_size);
				//cudaFree(cudaDeviceNode);
				run_simulation(dec_seq, cuda_rand_nums, cuda_win_increase, bd_size);
				// cudaFree(cuda_rand_nums);
				delete rand_nums;
				// cudaFree(cuda_win_increase);

				int* win_increase = new int[1];
				memcpy(win_increase, cuda_win_increase, sizeof(int));

				children[i]->wins += *win_increase;
				children[i]->sims += MAX_TRIAL_H;
				back_propagation(children[i], *win_increase, MAX_TRIAL_H);
				delete win_increase;
				if (checkAbort())break;
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

Deque<Point*>* generateAllMoves(CudaBoard* cur_board) {
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

CudaBoard* get_board(std::vector<Point> sequence, int bd_size) {
	CudaBoard *bd = new CudaBoard(bd_size);
	for (int i = 0; i < sequence.size(); i++) {
		bd->update_board(&sequence[i]);
	}

	return bd;
}

void deleteAllMoves(Deque<Point*>* moves) {
	Deque<Point*>::iterator it = moves->begin();

	for (; it != moves->end(); it++) {
		Point* p = *it;
		delete p;
	}
}

void Mcts::deleteAllMoves(std::vector<Point*> moves) {
	for (std::vector<Point*>::iterator it = moves.begin(); it != moves.end(); it++) {
		delete *it;
	}
}

// template <typename T>
// KernelArray<T> convertToKernel(thrust::device_vector<T>& dVec)
// {
// 	KernelArray<T> kArray;
// 	kArray._array = thrust::raw_pointer_cast(&dVec[0]);
// 	kArray._size  = (int) dVec.size();

// 	return kArray;
// }

