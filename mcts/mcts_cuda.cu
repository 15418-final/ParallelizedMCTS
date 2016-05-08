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
#include <stdint.h>

#include "mcts.h"
#include "CudaGo.h"
#include "deque.h"
#include "point.h"

#define BILLION 1000000000L
#define MILLION 1000000.0
//Exploration parameter
double C = 1.4;
double EPSILON = 10e-6;

__constant__ int MAX_TRIAL_H = 1;
#define MAX_STEP 300 // avoid repeat game
__constant__ double CLOCK_RATE = 1215500.0; // For tesla K40

int MAX_TRIAL = 1;
#define MAX_GAME_TIME_9_9 1000.0
double MAX_GAME_TIME_11_11 = 4000.0;

static int grid_dim = 2880;
static int block_dim = 1;
static int THREADS_NUM = grid_dim * block_dim;
#define CPU_THREADS_NUM 59

bool checkAbort();
__device__ bool checkAbortCuda(long long int elapse, double timeLeft);
__global__ void run_simulation(int incre, int total, int* iarray, int* jarray, int* len, double* win_increase,
                               int* step, double* sim, int bd_size, unsigned int seed, double time);
__device__ __host__ Point* createPoints(int bd_size);
void* run_simulation_thread(void *arg);
void get_sequence(TreeNode* node, int* len, int* iarray, int*jarray);

void memoryUsage();

Point Mcts::run() {
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

	return best->get_sequence().back();
}

TreeNode* Mcts::selection(TreeNode* node) {
	std::cout << "selection begin" << std::endl;
	double maxv = -1.0;
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

__global__ void run_simulation(int incre, int total, int* iarray, int* jarray, int* len, double* win_increase,
                               int* step, double* sim, int bd_size, unsigned int seed, double time) {
	long long int start_game = clock64();
	int index = blockIdx.x;
	win_increase[index] = 0.0;
	step[index] = 0;
	sim[index] = 0;
	bool abort = false;

	curandState_t state;
	curand_init(seed + index, 0, 0, &state);

	CudaBoard board(bd_size);

	int id = index / incre;
	if (id >= total) id = total - 1;
	int p = 0;
	for (int i = 0; i < id; i++) {
		p += len[i];
	}
	for (int i = p; i < p + len[id]; i++) {
		board.update_board(Point(iarray[i], jarray[i]));
	}
	COLOR player = board.ToPlay();

	while (step[index] < MAX_STEP) {
		Point move = board.get_next_moves_device(curand_uniform(&state));
		if (move.i < 0) {
			break;
		}
		board.update_board(move);
		if ((clock64() - start_game) / CLOCK_RATE > time) {
			abort = true;
			break;
		}
		if (index == 0) {
			printf("time elapse:%f\n", (clock64() - start_game) / CLOCK_RATE);
		}
		step[index]++;
	}

	int score = board.score();
	if ((score > 0 && player == BLACK)
	        || (score < 0 && player == WHITE)) {
		if (abort) {
			win_increase[index] += (double)step[index] / MAX_STEP;
		} else {
			win_increase[index]++;
		}
	}

	if (abort) {
		sim[index] += (double) step[index] / MAX_STEP;
	} else {
		sim[index]++;
	}

//	if (index == 0) {printf("time:%f, step: %d\n", (clock64() - start_game) / CLOCK_RATE, step[index]);}
}

void* run_simulation_thread(void *arg) {
	thread_arg* a = static_cast<thread_arg*> (arg);
	int len = a->len;
	double timeLeft = a->time;
	int cur_step = 0;
	a->sim = 0.0;
	a->win = 0.0;
	bool abort = false;
	COLOR player;
	clock_t start = clock();
	CudaBoard* board;
	srand (time(NULL));

	while (true) {
		board =  new CudaBoard(a->bd_size);
		for (int i = 0; i < len; i++) {
			board->update_board(a->seq[i]);
		}
		player = board->ToPlay();
		if ((1000.0 * (clock() - start) / CLOCKS_PER_SEC) > timeLeft) break;

		cur_step = 0;
		while (cur_step < MAX_STEP) {
			std::vector<Point> moves = board->get_next_moves_host();
			if (moves.size() == 0) {
				break;
			}
			board->update_board(moves[rand() % moves.size()]);
			if ((1000.0 * (clock() - start) / CLOCKS_PER_SEC) > timeLeft) {
				abort = true;
				break;
			}
			cur_step++;
		}

		int score = board->score();
		if ((score > 0 && player == BLACK)
		        || (score < 0 && player == WHITE)) {
			if (abort) {
				a->win += (double)cur_step / MAX_STEP;
			} else {
				a->win++;
			}
		}

		if (abort) {
			a->sim += (double)cur_step / MAX_STEP;
		} else {
			a->sim++;
		}
		if ((MAX_GAME_TIME_9_9 * (clock() - start) / CLOCKS_PER_SEC) > timeLeft) break;
		delete board;
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

void Mcts::update(TreeNode* node, double* sim, double* win, int incre, int thread_num) {
	std::vector<TreeNode*> children = node->get_children();
	for (int i = 0; i < thread_num; i++) {
		int id = i / incre;
		if (id >= children.size()) id = children.size() - 1;
		back_propagation(children[id], win[id], sim[id]);
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
	int* c_i = new int[total * total];
	int* c_j = new int[total * total];
	int* cpu_len = new int[total];
	double* win_increase = new double[THREADS_NUM];
	double* sim_increase = new double[THREADS_NUM];
	int* step_increase = new int[THREADS_NUM];
	int* c_i_d; // device
	int* c_j_d; // device
	int* cuda_len;


	cudaEvent_t start_event, stop;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("clock :%d\n", prop.clockRate);

	cudaMalloc(&cuda_len, sizeof(int) * total);
	cudaMalloc(&c_i_d, sizeof(int)* total * total);
	cudaMalloc(&c_j_d, sizeof(int)* total * total);

	// pthread_t* tids = (pthread_t*)malloc(sizeof(pthread_t) * CPU_THREADS_NUM);
	// thread_arg* args = (thread_arg*)malloc(sizeof(thread_arg) * CPU_THREADS_NUM);
	// for (int ti = 0; ti < CPU_THREADS_NUM; ti++) {
	// 	args[ti].seq = (Point*)malloc(sizeof(Point) * 300);
	// }

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

			get_sequence(f, cpu_len, c_i, c_j);
			int csize = f->get_children().size();
			int incre = THREADS_NUM / csize;

			// double thread_sim = 0;
			// for (int ti = 0; ti < CPU_THREADS_NUM; ti++) {
			// 	args[ti].len = (children[i]->get_sequence()).size();
			// 	for (int pi = 0; pi < args[ti].len; pi++) {
			// 		args[ti].seq[pi] = (children[i]->get_sequence())[pi];
			// 	}
			// 	args[ti].time = MAX_GAME_TIME_9_9;
			// 	args[ti].bd_size = bd_size;
			// 	args[ti].tid = ti;
			// 	pthread_create(&tids[ti], NULL, run_simulation_thread, (void *)(&args[ti]));
			// }

			thrust::device_ptr<double> cuda_win_increase = thrust::device_malloc<double>(THREADS_NUM);
			thrust::device_ptr<double> cuda_sim = thrust::device_malloc<double>(THREADS_NUM);
			thrust::device_ptr<int> cuda_step = thrust::device_malloc<int>(THREADS_NUM);

			cudaMemcpy(c_i_d, c_i, sizeof(int)*total * total, cudaMemcpyHostToDevice);
			cudaMemcpy(c_j_d, c_j, sizeof(int)*total * total, cudaMemcpyHostToDevice);
			cudaMemcpy(cuda_len, cpu_len, sizeof(int)*total, cudaMemcpyHostToDevice);

			uint64_t diff;
			clock_gettime(CLOCK_REALTIME, &end);
			diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
			double timeLeft = maxTime - diff / MILLION;

			cudaEventRecord(start_event);
			run_simulation <<< grid_dim, block_dim >>> (incre, csize, c_i_d, c_j_d, cuda_len, cuda_win_increase.get(), cuda_step.get(), cuda_sim.get(),
			        bd_size, time(NULL), std::min(MAX_GAME_TIME_9_9, timeLeft));
			cudaEventRecord(stop);

			printf("return : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));

			// for (int ti = 0; ti < CPU_THREADS_NUM; ti++) {
			// 	pthread_join(tids[ti], NULL);
			// 	thread_sim += args[ti].sim;
			// }

			// printf("thread done, sim: %d\n", thread_sim);

			//memoryUsage();
			printf("THREADS_NUM:%d\n", THREADS_NUM);

			cudaMemcpy(win_increase, cuda_win_increase.get(), sizeof(double) * THREADS_NUM, cudaMemcpyDeviceToHost);
			cudaMemcpy(step_increase, cuda_step.get(), sizeof(int) * THREADS_NUM, cudaMemcpyDeviceToHost);
			cudaMemcpy(sim_increase, cuda_sim.get(), sizeof(double) * THREADS_NUM, cudaMemcpyDeviceToHost);

			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start_event, stop);

			double total_sim = 0.0;
			double total_win = 0.0;
			int total_step = 0;
			for (int i = 0; i < THREADS_NUM; i++) {
				total_sim += sim_increase[i];
				total_win += win_increase[i];
				total_step += step_increase[i];
			}
			printf("time measured in CPU: %lf\n", milliseconds);
			printf("win: %f\n", total_win);
			printf("step: %d\n", total_step);
			printf("gpu sim: %f\n", total_sim);
			//printf("gpu sim: %f, totoal:%f\n", total_sim[0], total_sim[0] + thread_sim);

			update(f, win_increase, sim_increase, incre, THREADS_NUM);

			thrust::device_free(cuda_win_increase);
			thrust::device_free(cuda_step);
			thrust::device_free(cuda_sim);
			if (checkAbort())break;
		}

		if (checkAbort()) break;
	}
	std::cout << "run_iteration end:" << std::endl;
	delete [] c_i;
	delete [] c_j;
	delete [] cpu_len;
	delete [] win_increase;
	delete [] sim_increase;
	delete [] step_increase;
}

void get_sequence(TreeNode* node, int* len, int* iarray, int*jarray) {
	std::vector<TreeNode*> children = node->get_children();
	int p = 0;
	for (size_t i = 0; i < children.size(); i++) {
		std::vector<Point> sequence = children[i]->get_sequence();
		len[i] = sequence.size();
		for (int it = 0; it < len[i]; it++) {
			iarray[it + p] = sequence[it].i;
			jarray[it + p] = sequence[it].j;
		}
		p += len[i];
	}
}

bool Mcts::checkAbort() {
	if (!abort) {
		uint64_t diff;
		clock_gettime(CLOCK_REALTIME, &end);
		diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
		abort = diff / MILLION > maxTime;
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

