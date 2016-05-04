#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <time.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include "mcts.h"
#include "CudaGo.h"
#include "deque.h"

//Exploration parameter
double C = 1.4;
double EPSILON = 10e-6;

int MAX_TRIAL_H = 50;


__device__ bool checkAbort();
__device__ Deque<Point*>* generateAllMoves(CudaBoard* cur_board);
__device__ void deleteAllMoves(Deque<Point*>* moves);
__global__ void run_simulation(int* iarray, int* jarray, int len, int* win_increase, int bd_size, int seed);

void memoryUsage();

Point Mcts::run() {
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
__global__ void run_simulation(int* iarray, int* jarray, int len, int* win_increase, int bd_size, unsigned int seed) {
	CudaBoard* board = new CudaBoard(bd_size);
	for (int i = 0; i < len; i++) {
		Point* p = new Point(iarray[i], jarray[i]);
		board->update_board(p);
		delete p;
	}
	*win_increase = 0;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	COLOR cur_player = board->ToPlay();
	curandState_t state;
	curand_init(seed ,0,0,&state);

	int time = 0;

		// bool timeout = false;
		CudaBoard* cur_board = new CudaBoard(*board);
		
		
		while (true) {
			Deque<Point*>* moves_vec = generateAllMoves(cur_board);
			if (cur_board->EndOfGame() || moves_vec->size() == 0) {
				break;
			}
			//why nxt_move length can be zero? what does endofgame do above?
			// std::cout << "moves_vec length:" << moves_vec->Length() << std::endl;
			Point* nxt_move = (*moves_vec)[curand(&state) % moves_vec->size()];
			cur_board->update_board(nxt_move);
			deleteAllMoves(moves_vec);
			delete moves_vec;
			time++;
		}
		
			int score = cur_board->score(); // Komi set to 0
			if ((score > 0 && cur_player == BLACK)
			        || (score < 0 && cur_player == WHITE)) {
				(*win_increase)++;
			}
		
		printf("run simulation done\n");
		delete cur_board;
	
	*win_increase = time;
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
	std::cout<<"moves generated:"<< moves_vec.size() <<std::endl;
	while (moves_vec.size() > 0) {
		Point* nxt_move = moves_vec.back();
		node->add_children(new TreeNode(node->get_sequence(), *nxt_move));
		moves_vec.pop_back();
		delete nxt_move;
	}
	std::cout<<"children add done"<<std::endl;
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
			std::cout<<"expand f end:"<<f<<std::endl;

			std::vector<TreeNode*> children = f->get_children();
			for (size_t i = 0; i < children.size(); i++) {
				// TreeNode* cudaDeviceNode = NULL;
				// int* cuda_win_increase = NULL;
				// // Use cuda to parallelize
				// cudaMalloc((void **)&cudaDeviceNode, sizeof(*children[i]));
				// cudaMalloc((void **)&cuda_win_increase, sizeof(int));
				// cudaMemcpy(cudaDeviceNode, children[i], sizeof(*children[i]), cudaMemcpyHostToDevice);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				int* cuda_win_increase = NULL;
				cudaMalloc((void **)&cuda_win_increase, sizeof(int));
				std::cout<<"Cuda malloc done"<<std::endl;

				std::vector<Point> sequence = children[i]->get_sequence();
				int len = sequence.size();
				int* c_i = new int[len];
				int* c_j = new int[len];
				int* c_i_d; // device
				int* c_j_d; // device
				for (int it = 0; it < len; it++) {
					c_i[it] = sequence[it].i;
					c_j[it] = sequence[it].j;
				}

				cudaMalloc(&c_i_d, sizeof(int)*len);
    			cudaMalloc(&c_j_d, sizeof(int)*len);
    			cudaMemcpy(c_i_d, c_i, sizeof(int)*len, cudaMemcpyHostToDevice); 
    			cudaMemcpy(c_j_d, c_j, sizeof(int)*len, cudaMemcpyHostToDevice); 
				
				CudaBoard* board = get_board(sequence, bd_size);
				board->print_board();

				std::cout<<"ready to run cuda code run_simulation()"<<std::endl;
				cudaEventRecord(start);
				run_simulation<<<1,1>>>(c_i_d, c_j_d, len, cuda_win_increase, bd_size, time(NULL));
				cudaEventRecord(stop);
				printf("return : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
				
				memoryUsage();

				//cudaFree(cudaDeviceNode);
				cudaDeviceSynchronize();
				int* win_increase = new int[1];
				cudaMemcpy(win_increase, cuda_win_increase, sizeof(int), cudaMemcpyDeviceToHost);
				
				cudaEventSynchronize(stop);
				float milliseconds = 0;
				cudaEventElapsedTime(&milliseconds, start, stop);
				

				printf("time: %f\n", milliseconds);
				printf("win: %d\n", *win_increase);
			
				cudaFree(c_i_d);
				cudaFree(c_j_d);

				children[i]->wins += *win_increase;
				children[i]->sims += MAX_TRIAL_H;
				//printf("win:%d, sims:%d\n", children[i]->wins, children[i]->sims);
				back_propagation(children[i], *win_increase, MAX_TRIAL_H);
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


__device__ void deleteAllMoves(Deque<Point*>* moves) {
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

void memoryUsage() {
	size_t free_byte ;

        size_t total_byte ;

          cudaMemGetInfo( &free_byte, &total_byte ) ;

  


        double free_db = (double)free_byte ;

        double total_db = (double)total_byte ;

        double used_db = total_db - free_db ;

        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

