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
#include "point.h"

//Exploration parameter
double C = 1.4;
double EPSILON = 10e-6;

__constant__ int MAX_TRIAL_H = 5;
int MAX_TRIAL = 5;

static int grid_dim = 8;
static int block_dim = 8;
static int THREADS_NUM = grid_dim * block_dim;


int wrapSequence(std::vector<TreeNode*> children, Point* &existingPath, Point* &allNxtMoves);
bool checkAbort();
__device__ bool checkAbortCuda(bool* abort, clock_t startTime, double timeLeft);
__global__ void run_simulation(Point* existingPath, int pathLen, Point* allNxtMoves, int len, int* win_increase, Point* parray, int bd_size, unsigned int seed);
__device__ __host__ Point* createPoints(int bd_size);

void memoryUsage();

Point Mcts::run() {
	// mcts_timer.Start();
	size_t heapszie = 128 * 1024 * 1024;
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

__global__ void run_simulation(Point* existingPath, int pathLen, Point* allNxtMoves, int len, int* win_increase, Point* parray, double timeLeft, int bd_size, unsigned int seed) {
	// TODO: use shared memory for point
	// __shared__ Point* point;
	
	// if (threadIdx.x == 0) {
	// 	memcpy(point, parray, sizeof(Point)*(bd_size+2)*(bd_size+2));
	// }
	// __syncthreads();

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// __shared__ Point* sequence;
	// if(blockIdx.x < len && threadIdx.x == 0){
	// 	printf("children size:%d\n", len);
	// 	printf("blockIdx:%d\n", blockIdx.x);
	// 	printf("in global: %d,%d\n", allNxtMoves[blockIdx.x].i, allNxtMoves[blockIdx.x].j);
		
	// 	printf("done\n");
	// }
	win_increase[blockIdx.x] = 0;
	// __syncthreads();

	clock_t cudaStartTime = clock();
	CudaBoard* initBoard = new CudaBoard(bd_size);
	COLOR player = initBoard->ToPlay();
	curandState_t state;
	curand_init(seed + index, 0, 0, &state);
	for (int i = 0; i < pathLen; i++) {
		initBoard->update_board(parray[existingPath[i].i*(bd_size+2) + existingPath[i].j], parray);
	}
	for(int i = 0; i < MAX_TRIAL_H; i++){
		CudaBoard board = *initBoard;
		bool abort = false;
		int times = 0;
		int step = 0;
		
		board.update_board(parray[allNxtMoves[blockIdx.x].i * (bd_size+2) + allNxtMoves[blockIdx.x].j], parray);
	
		while (true && step < 300) {
			Deque<Point>* moves = board.get_next_moves_device(parray);
			if (moves->size() == 0) {
				break;
			}
			Point nxt_move = (*moves)[curand(&state) % moves->size()];
			board.update_board(nxt_move, parray);
			step++;
			// if(checkAbortCuda(&abort, cudaStartTime, timeLeft))break;
		}
		times++;
			// printf("time used for one game:%lf\n", 1000.0 * (std::clock() - ttime) / CLOCKS_PER_SEC);
			// if(checkAbortCuda(&abort, cudaStartTime, timeLeft))break;
		int score = board.score(); // Komi set to 0

		if ((score > 0 && player == BLACK)
		        || (score < 0 && player == WHITE)) {
			atomicInc((unsigned int*)&(win_increase[blockIdx.x]), 10000000);
		}
		printf("win in block %d: %d\n", blockIdx.x, win_increase[blockIdx.x]);
	
	}
	delete initBoard;
	// if(index == 0) 
	// 	printf("num of trial done:%d\n",times);
	// if(index == 0) 	printf("time cp7:%ld\n", (std::clock()) / CLOCKS_PER_SEC);
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

	std::vector<Point> moves_vec = generateAllMoves(cur_board);
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

	Point* points = createPoints(bd_size);

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
			int len = f->get_children().size();
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			Point* cuda_points = NULL;
			thrust::device_ptr<int> cuda_win_increase = thrust::device_malloc<int>(grid_dim);

			cudaMalloc(&cuda_points, sizeof(Point) * (bd_size + 2) * (bd_size + 2));
			// std::cout << "Cuda malloc done" << std::endl;

			cudaMemcpy(cuda_points, points, sizeof(Point) * (bd_size + 2) * (bd_size + 2), cudaMemcpyHostToDevice);
			
			double timeLeft = maxTime - 1000.0*(std::clock() - startTime)/double(CLOCKS_PER_SEC);
				// printf("startTime before kernel:%ld\n",startTime);
				// printf("current clock time:%ld\n",clock());
				// printf("CLOCKS_PER_SEC:%ld\n",CLOCKS_PER_SEC);
				// printf("timeLeft before kernel:%lf\n", timeLeft);
				// std::cout << "ready to run cuda code run_simulation()" << std::endl;
			cudaEventRecord(start);

			Point* existingPath = NULL;
			Point* allNxtMoves = NULL;
			int pathLen = wrapSequence(f->get_children(), existingPath, allNxtMoves);

			run_simulation <<< grid_dim, block_dim >>> (existingPath, pathLen, allNxtMoves, len, cuda_win_increase.get(), cuda_points, timeLeft, bd_size, time(NULL));

			cudaEventRecord(stop);
			printf("return : %s\n", cudaGetErrorString(cudaDeviceSynchronize()));

			cudaDeviceSynchronize();
			memoryUsage();

    			
			int* win_increase = new int[grid_dim];
    		cudaMemcpy(win_increase, cuda_win_increase.get(), sizeof(int)*grid_dim, cudaMemcpyDeviceToHost);


			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);

			printf("time measured in CPU: %lf\n", milliseconds);
			// for(int i = 0; i < grid_dim; i++){
			// 	printf("win[i]: %d\n", win_increase[i]);
			// }
			

			cudaDeviceReset();
			for(int i = 0; i < grid_dim; i++){
				f->get_children()[i]->wins += win_increase[i];
				f->get_children()[i]->sims += MAX_TRIAL*grid_dim;
				back_propagation(f->get_children()[i], f->get_children()[i]->wins, f->get_children()[i]->sims);
			}

			
			delete [] win_increase;
			if (checkAbort())break;
		}
		if (checkAbort()) break;
	}
	std::cout << "run_iteration end:" << std::endl;
}

__device__ bool checkAbortCuda(bool* abort, clock_t cudaStartTime, double timeLeft){
	if (!(*abort)) {
		*abort = 1000.0 * (std::clock() - cudaStartTime) / CLOCKS_PER_SEC > timeLeft;
	}

	// if(*abort) printf("is aborted in device, timeLeft:%lf, startTime:%d\n",timeLeft, cudaStartTime);
	// else{
	// 	printf("not aborted yet. lhs:%lf\n", 1000.0 * (std::clock() - cudaStartTime) / CLOCKS_PER_SEC);
	// }
	return *abort;
}

bool Mcts::checkAbort() {
	if (!abort) {
		abort = 1000.0 * (std::clock() - startTime) / CLOCKS_PER_SEC > maxTime;
	}
	if(abort) printf("is aborted in host\n");
	return abort;
}

std::vector<Point> Mcts::generateAllMoves(CudaBoard* cur_board) {
	Point* point = createPoints(bd_size);
	std::vector<Point> moves_vec = cur_board->get_next_moves_host(point);
	int len = moves_vec.size();

	/* NOTE: point has not been freed yet !!!!!*/

	return moves_vec;
}

CudaBoard* Mcts::get_board(std::vector<Point> sequence, int bd_size) {
	Point* point = createPoints(bd_size);
	CudaBoard* bd = new CudaBoard(bd_size);
	for (std::vector<Point>::iterator it = sequence.begin(); it != sequence.end(); it++) {
		bd->update_board((*it), point);
	}
	return bd;
}



void Mcts::deleteAllMoves(std::vector<Point*> moves) {
	for (std::vector<Point*>::iterator it = moves.begin(); it != moves.end(); it++) {
		delete *it;
	}
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

Deque<Point> vec2deq(std::vector<Point> vp){
	Deque<Point> rst;
	for(int i = 0; i < vp.size(); i++){
		rst.push_back(vp[i]);
	}
	return rst;
}

int wrapSequence(std::vector<TreeNode*> children, Point* &existingPath, Point* &allNxtMoves){
	cudaMalloc((void**)&existingPath, children[0]->get_sequence().size()*sizeof(Point));
	cudaMalloc((void**)&allNxtMoves, sizeof(Point)*children.size());

	int commonPathLen = children[0]->get_sequence().size()-1;
	Point* temp = (Point*)malloc(sizeof(Point)*commonPathLen);
	for(int i = 0; i < commonPathLen; i++){
		memcpy(temp+i, &(children[0]->get_sequence()[i]), sizeof(Point));
	}
	cudaMemcpy(existingPath, temp, sizeof(Point)*commonPathLen, cudaMemcpyHostToDevice);

	temp = (Point*)realloc(temp, sizeof(Point) * children.size());


	for(int i = 0; i < children.size(); i++){
		memcpy(temp+i, &(children[i]->get_sequence().back()), sizeof(Point));
	}
	cudaMemcpy(allNxtMoves, temp, sizeof(Point)*children.size(), cudaMemcpyHostToDevice);
	delete temp;
	return commonPathLen;
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

