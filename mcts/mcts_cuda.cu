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

__constant__ int MAX_TRIAL_H = 4;
int MAX_TRIAL = 5;

static int grid_dim = 4;
static int block_dim = 1;
static int THREADS_NUM = grid_dim * block_dim;

bool checkAbort();
__device__ bool checkAbortCuda(bool* abort, clock_t startTime, double timeLeft);
__global__ void run_simulation(int* iarray, int* jarray, int len, int* win_increase, double timeLeft, int bd_size, unsigned int seed);
__device__ __host__ Point*** createPoints(int bd_size);
__device__ __host__ void deletePoints(Point*** point, int bd_size);
__device__ void deleteAllMoves(Deque<Point*>* moves);

void memoryUsage();

Point Mcts::run() {
	// mcts_timer.Start();
	int iter = 0;
	while (iter < 1) {
		run_iteration(root);
		if (checkAbort()) break;
		iter++;
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
__global__ void run_simulation(int* iarray, int* jarray, int len, int* win_increase, double timeLeft, int bd_size, unsigned int seed) {
	// TODO: use shared memory for point
	clock_t very_start = 0;

	__shared__ Point*** globalPoints;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadIdx.x == 0){
		globalPoints = createPoints(bd_size);
	}
	if(index == 0)
	printf("time cp1:%lld\n", (std::clock() - very_start) / CLOCKS_PER_SEC);
	__syncthreads();
	extern __shared__ int wins[]; //size declared win kernel is called.
	bool abort = false;
	win_increase[index] = 0;
	clock_t cudaStartTime = std::clock();
	int times = 0;
	for(int trial = 0; trial < MAX_TRIAL_H; trial++){
		CudaBoard* board = new CudaBoard(bd_size);
		if(index == 0)
		printf("time cp2:%ld\n", (std::clock() - very_start) / CLOCKS_PER_SEC);

		for (int i = 0; i < len; i++) {
			board->update_board(globalPoints[iarray[i]][jarray[i]], globalPoints);
		}
		if(index == 0)
		printf("time cp3:%ld\n", (std::clock() - very_start) / CLOCKS_PER_SEC);

		// for (int i = 0; i < 100; i++) {
		// 	Deque<Point*>* moves = board->get_next_moves_device(globalPoints);
		// 	board->update_board(moves->front(), globalPoints);
		// }

		COLOR player = board->ToPlay();
		curandState_t state;
		curand_init(seed + index, 0, 0, &state);

		// bool timeout = false;
		*win_increase = 0;
		int step = 0;
		clock_t ttime = std::clock();
		if(index == 0)
		printf("time cp4:%lld\n", (std::clock() - very_start) / CLOCKS_PER_SEC);
		while (true) {
			Deque<Point*>* moves = board->get_next_moves_device(globalPoints);
			if (moves->size() == 0) {
				break;
			}
			
			Point* nxt_move = (*moves)[curand(&state) % moves->size()];
			//Point* nxt_move = moves->front();
			board->update_board(nxt_move, globalPoints);
			step++;
			if(checkAbortCuda(&abort, cudaStartTime, timeLeft))break; 
		}
		if(index == 0){
			printf("time cp5:%ld\n", (std::clock() - very_start) / CLOCKS_PER_SEC);
		}
		

		// printf("id:%d, step:%d, timeleft:%lf\n", index, step, timeLeft);
		times++;
		// printf("time used for one game:%lf\n", 1000.0 * (std::clock() - ttime) / CLOCKS_PER_SEC);
		if(checkAbortCuda(&abort, cudaStartTime, timeLeft))break;
		int score = board->score(); // Komi set to 0
		if(index == 0)
		printf("time cp6:%ld\n", (std::clock() - very_start) / CLOCKS_PER_SEC);
		if ((score > 0 && player == BLACK)
		        || (score < 0 && player == WHITE)) {
			win_increase[index]++;
		}

		delete board;
	}
	if(index == 0) 
		printf("num of trial done:%d\n",times);
	if(index == 0) 	printf("time cp7:%ld\n", (std::clock() - very_start) / CLOCKS_PER_SEC);
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
		delete nxt_move;
	}
	std::cout << "children add done" << std::endl;
	deleteAllMoves(moves_vec);
	delete cur_board;

	std::cout << "expand end with children num:" << node->get_children().size() << std::endl;
}

void Mcts::run_iteration(TreeNode* node) {
	std::stack<TreeNode*> S;
	S.push(node);

	int total = bd_size * bd_size;
	int* c_i = new int[total];
	int* c_j = new int[total];
	int* c_i_d; // device
	int* c_j_d; // device

	std::cout << "run_iteration start:" << std::endl;

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

			std::vector<TreeNode*> children = f->get_children();
			for (size_t i = 0; i < children.size(); i++) {
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);


				// cudaMemset(cuda_win_increase, 0, sizeof(int)*THREADS_NUM);
				cudaMalloc(&c_i_d, sizeof(int)*total);
				cudaMalloc(&c_j_d, sizeof(int)*total);
				std::cout << "Cuda malloc done" << std::endl;

				std::vector<Point> sequence = children[i]->get_sequence();
				int len = sequence.size();
    			thrust::device_ptr<int> cuda_win_increase = thrust::device_malloc<int>(THREADS_NUM);

				for (int it = 0; it < len; it++) {
					c_i[it] = sequence[it].i;
					c_j[it] = sequence[it].j;
				}

				cudaMemcpy(c_i_d, c_i, sizeof(int)*len, cudaMemcpyHostToDevice);
				cudaMemcpy(c_j_d, c_j, sizeof(int)*len, cudaMemcpyHostToDevice);

				CudaBoard* board = get_board(sequence, bd_size);
				board->print_board();

				double timeLeft = maxTime - 1000.0*(std::clock() - startTime)/double(CLOCKS_PER_SEC);
				// printf("startTime before kernel:%ld\n",startTime);
				// printf("current clock time:%ld\n",clock());
				// printf("CLOCKS_PER_SEC:%ld\n",CLOCKS_PER_SEC);
				// printf("timeLeft before kernel:%lf\n", timeLeft);
				// std::cout << "ready to run cuda code run_simulation()" << std::endl;
				cudaEventRecord(start);
				run_simulation <<<grid_dim, block_dim>>>(c_i_d, c_j_d, len, cuda_win_increase.get(), timeLeft, bd_size, time(NULL));
				cudaEventRecord(stop);
				printf("return : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));

				memoryUsage();
				cudaDeviceSynchronize();

    			thrust::device_ptr<int> d_output = thrust::device_malloc<int>(THREADS_NUM);
    
    			thrust::exclusive_scan(cuda_win_increase, cuda_win_increase + THREADS_NUM, d_output);

    			cudaDeviceSynchronize();
    			
				int* win_increase = new int[1];
    			cudaMemcpy(win_increase, d_output.get()+THREADS_NUM-1, sizeof(int), cudaMemcpyDeviceToHost);
			    thrust::device_free(d_output);

				cudaEventSynchronize(stop);
				float milliseconds = 0;
				cudaEventElapsedTime(&milliseconds, start, stop);


				printf("time measured in CPU: %lf\n", milliseconds);
				// for(int w = 0 ; w < THREADS_NUM; w++){
				// 	printf("win[i]: %d\n", win_increase[w]);
				// }
				

				cudaDeviceReset();
				children[i]->wins += win_increase[0];
				children[i]->sims += MAX_TRIAL*THREADS_NUM;
				printf("win:%d, sims:%d\n", children[i]->wins, children[i]->sims);
				back_propagation(children[i], win_increase[0], MAX_TRIAL);
				delete win_increase;
				// if (checkAbort())break;

			}
		}
		if (checkAbort()) break;
	}
	std::cout << "run_iteration end:" << std::endl;
}

__device__ bool checkAbortCuda(bool* abort, clock_t cudaStartTime, double timeLeft){
	if (!(*abort)) {
		*abort = 1000.0 * (std::clock() - cudaStartTime) / CLOCKS_PER_SEC > timeLeft;
	}

	if(*abort) printf("is aborted in device, timeLeft:%lf, startTime:%d\n",timeLeft, cudaStartTime);
	else{
		printf("not aborted yet. lhs:%lf\n", 1000.0 * (std::clock() - cudaStartTime) / CLOCKS_PER_SEC);
	}
	return *abort;
}

bool Mcts::checkAbort() {
	if (!abort) {
		abort = 1000.0 * (std::clock() - startTime) / CLOCKS_PER_SEC > maxTime;
	}
	if(abort) printf("is aborted in host\n");
	return abort;
}

std::vector<Point*> Mcts::generateAllMoves(CudaBoard* cur_board) {
	Point*** point = createPoints(bd_size);

	std::vector<Point*> moves_vec = cur_board->get_next_moves_host(point);
	int len = moves_vec.size();

	/* NOTE: point has not been freed yet !!!!!*/

	return moves_vec;
}

CudaBoard* Mcts::get_board(std::vector<Point> sequence, int bd_size) {
	Point*** point = createPoints(bd_size);
	CudaBoard* bd = new CudaBoard(bd_size);
	for (std::vector<Point>::iterator it = sequence.begin(); it != sequence.end(); it++) {
		bd->update_board(&(*it), point);
	}
	deletePoints(point, bd_size);
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

__device__ __host__ Point*** createPoints(int bd_size) {
	int len = bd_size + 2;
	Point*** point = static_cast<Point***> (malloc(sizeof(Point*) * len));
	for (int i = 0; i < len; i++) {
		point[i] = static_cast<Point**> (malloc(sizeof(Point*) * len));
		for (int j = 0; j < len; j++) {
			point[i][j] = (Point*)malloc(sizeof(Point));
			point[i][j]->i = i;
			point[i][j]->j = j;
		}
	}
	return point;
}

__device__ __host__ void deletePoints(Point*** point, int bd_size) {
	for (int i = 0; i < bd_size + 2; i++) {
		for (int j = 0; j < bd_size + 2; j++) {
			delete point[i][j];
		}
		free(point[i]);
	}
	free(point);
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

