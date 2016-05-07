#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <time.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/extrema.h>

#include "mcts.h"
#include "CudaGo.h"
#include "deque.h"
#include "point.h"

//Exploration parameter
double C = 1.4;
double EPSILON = 10e-6;

__constant__ int MAX_TRIAL_H = 1;
__constant__ int MAX_STEP = 300; // avoid repeat game
__constant__ double CLOCK_RATE = 745000.0; // For tesla K40

int MAX_TRIAL = 1;
double MAX_GAME_TIME_9_9 = 1500.0;

static int grid_dim = 1024;
static int block_dim = 1;
static int THREADS_NUM = grid_dim * block_dim;

bool checkAbort();
__device__ bool checkAbortCuda(long long int elapse, double timeLeft);
__global__ void run_simulation(int* iarray, int* jarray, int len, double* win_increase, int* step, double* sim, double* cuda_time, int bd_size, unsigned int seed, double time);
__device__ __host__ Point* createPoints(int bd_size);

void memoryUsage();

Point Mcts::run() {
	// mcts_timer.Start();
	size_t heapszie = 1024 * 1024 * 1024;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapszie);

	while (true) {
		run_iteration(root);
		if (checkAbort()) break;
	}
	double maxv = -1.0;
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

__global__ void run_simulation(int* iarray, int* jarray, int len, double* win_increase, int* step, double* sim, double* cuda_time, int bd_size, unsigned int seed, double time) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	win_increase[index] = 0.0;
	step[index] = 0;
	cuda_time[index] = 0;
	sim[index] = 0;
	double timeLeft = time;

	long long start;
	long long end;
	long long start_game;
	long long end_game;
	curandState_t state;
	CudaBoard board(bd_size);

	start_game = clock64();
	while (true) {
		curand_init(seed + index + step[index], 0, 0, &state);
		start = clock64();
		for (int i = 0; i < len; i++) {
			board.update_board(Point(iarray[i], jarray[i]));
		}
		COLOR player = board.ToPlay();

		end = clock64();
		timeLeft -= (end - start) / CLOCK_RATE;
		if (timeLeft <= 0.0) break;

		int cur_step = 0;
		while (cur_step < MAX_STEP) {
			start = clock64();
			Point move = board.get_next_moves_device(curand_uniform(&state));
			if (move.i < 0) {
				break;
			}
			board.update_board(move);
			step[index]++;
			cur_step++;
			end = clock64();
			timeLeft -= (end - start) / CLOCK_RATE;
			if (timeLeft <= 0) break;
		}


		int score = board.score(); 
		if ((score > 0 && player == BLACK)
		        || (score < 0 && player == WHITE)) {
			if (timeLeft <= 0) {
			  win_increase[index] += cur_step / MAX_STEP;
			} else {
			  win_increase[index] += 1.0;
			}
			
		}
		if (timeLeft <= 0) {
			sim[index] += cur_step / MAX_STEP;
		} else {
			sim[index] += 1.0;
		}

		if (timeLeft <= 0) break;
		board.clear();
	}
	end_game = clock64();
	cuda_time[index] = (end_game - start_game) / CLOCK_RATE;
if (index % 100 == 0)
	printf("index:%d, win:%f, step:%d, sim:%f, time:%f\n", index, win_increase[index], step[index], sim[index], cuda_time[index]);
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

	std::vector<Point> moves_vec = cur_board->get_next_moves_host();
	while (moves_vec.size() > 0) {
		Point nxt_move = moves_vec.back();
		node->add_children(new TreeNode(node->get_sequence(), nxt_move));
		moves_vec.pop_back();
	}
	delete cur_board;

	std::cout << "expand end with children num:" << node->get_children().size() << std::endl;
}

void Mcts::run_iteration(TreeNode* node) {
	std::stack<TreeNode*> S;
	S.push(node);

	int total = bd_size * bd_size;
	int* c_i = new int[total];
	int* c_j = new int[total];
	double* win_increase = new double[1];
	double* total_sim = new double[1];
	int* total_step = new int[1];
	int* c_i_d; // device
	int* c_j_d; // device
	
	double* cpu_time = new double[THREADS_NUM];

	std::cout << "run_iteration start:" << std::endl;

	while (!S.empty()) {
		TreeNode* f = S.top();
		S.pop();
		if (!f->is_expandable()) {
			S.push(selection(f));
		} else {
			// expand current node, run expansion and simulation
			f->set_expandable(false);
			expand(f);

			std::vector<TreeNode*> children = f->get_children();
			for (size_t i = 0; i < children.size(); i++) {
				std::vector<Point> sequence = children[i]->get_sequence();
				int len = sequence.size();

				for (int it = 0; it < len; it++) {
					c_i[it] = sequence[it].i;
					c_j[it] = sequence[it].j;
				}

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaMalloc(&c_i_d, sizeof(int)*len);
				cudaMalloc(&c_j_d, sizeof(int)*len);
				thrust::device_ptr<double> cuda_win_increase = thrust::device_malloc<double>(THREADS_NUM);
				thrust::device_ptr<double> cuda_sim = thrust::device_malloc<double>(THREADS_NUM);
				thrust::device_ptr<int> cuda_step = thrust::device_malloc<int>(THREADS_NUM);
				thrust::device_ptr<double> cuda_time = thrust::device_malloc<double>(THREADS_NUM);

				cudaMemcpy(c_i_d, c_i, sizeof(int)*len, cudaMemcpyHostToDevice);
				cudaMemcpy(c_j_d, c_j, sizeof(int)*len, cudaMemcpyHostToDevice);

				CudaBoard* board = get_board(sequence, bd_size);
				board->print_board();

				double timeLeft = maxTime - 1000.0 * (std::clock() - startTime) / double(CLOCKS_PER_SEC);

				cudaEventRecord(start);
				run_simulation <<< grid_dim, block_dim >>> (c_i_d, c_j_d, len, cuda_win_increase.get(), cuda_step.get(), cuda_sim.get(), 
														    cuda_time.get(), bd_size, time(NULL), std::min(timeLeft,MAX_GAME_TIME_9_9));
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				printf("return : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));

				memoryUsage();
				printf("THREADS_NUM:%d\n", THREADS_NUM);

				thrust::inclusive_scan(cuda_win_increase, cuda_win_increase + THREADS_NUM, cuda_win_increase);
				thrust::inclusive_scan(cuda_step, cuda_step + THREADS_NUM, cuda_step);
				thrust::inclusive_scan(cuda_sim, cuda_sim + THREADS_NUM, cuda_sim);
				cudaDeviceSynchronize();


				cudaMemcpy(win_increase, cuda_win_increase.get() + THREADS_NUM - 1, sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(total_step, cuda_step.get() + THREADS_NUM - 1, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(cpu_time, cuda_time.get(), sizeof(double) * THREADS_NUM, cudaMemcpyDeviceToHost);
				cudaMemcpy(total_sim, cuda_sim.get() + THREADS_NUM - 1, sizeof(double), cudaMemcpyDeviceToHost);

				float milliseconds = 0;
				cudaEventElapsedTime(&milliseconds, start, stop);

				double max_time = -1.0;
				double total_time = 0.0;
				for (int k = 0; k < THREADS_NUM; k++) {
					if (max_time < cpu_time[k]) max_time = cpu_time[k];
					total_time += cpu_time[k];
				}

				printf("time measured in CPU: %lf\n", milliseconds);
				printf("win: %f\n", win_increase[0]);
				printf("step: %d\n", total_step[0]);
				printf("sim: %f\n", total_sim[0]);
				printf("max time:%f, average:%f\n", max_time, total_time / THREADS_NUM);

				cudaDeviceReset();

				children[i]->wins += win_increase[0];
				children[i]->sims += MAX_TRIAL * THREADS_NUM;

				back_propagation(children[i], children[i]->wins, children[i]->sims);
				if (checkAbort())break;
			}
		}
		if (checkAbort()) break;
	}
	std::cout << "run_iteration end:" << std::endl;
	delete [] c_i;
	delete [] c_j;
}

bool Mcts::checkAbort() {
	if (!abort) {
		abort = 1000.0 * (std::clock() - startTime) / CLOCKS_PER_SEC > maxTime;
	}
	if (abort) printf("is aborted in host\n");
	return abort;
}

CudaBoard* Mcts::get_board(std::vector<Point> sequence, int bd_size) {
	CudaBoard* bd = new CudaBoard(bd_size);
	for (std::vector<Point>::iterator it = sequence.begin(); it != sequence.end(); it++) {
		bd->update_board((*it));
	}
	return bd;
}

__device__ __host__ Point* createPoints(int bd_size) {
	int len = bd_size + 2;
	Point* point = static_cast<Point*> (malloc(sizeof(Point) * len * len));
	for (int i = 0; i < len; i++) {
		for (int j = 0; j < len; j++) {
			point[i * len + j] = Point(i, j);
		}
	}
	return point;
}

void memoryUsage() {
	size_t free_byte ;

	size_t total_byte ;

	cudaMemGetInfo( &free_byte, &total_byte ) ;

	double free_db = (double)free_byte ;

	double total_db = (double)total_byte ;

	double used_db = total_db - free_db ;

	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

	       used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

