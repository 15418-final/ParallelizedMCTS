#include <cstdio>
#include <ctime>
#include <omp.h>
#include "mcts.h"

//Exploration parameter
const double C = 1.4;
const double EPSILON = 10e-6;
const int MAX_TRIAL = 100;

const int THREAD_NUM = 16;

static int totalSimu = 0;
Point* createPoints(int bd_size);
Point Mcts::run() {
	std::cout << "maxTime:" << maxTime << std::endl;
	std::cout << "THREAD_NUM:" << THREAD_NUM << std::endl;
	gettimeofday(&startTime, NULL);
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
	// if (best == NULL) {
	// 	return NULL;
	// }
	std::cout << "Total simulation runs:" << totalSimu << std::endl;
	printf("decision move: %d,%d\n", best->get_sequence().back().i, best->get_sequence().back().j);
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
int Mcts::run_simulation(TreeNode* node) {
	GoBoard* board = get_board(node->get_sequence());
	Point* points = createPoints(bd_size);
	COLOR cur_player = board->ToPlay();
	// printf("last pos:%d,%d\n", node->get_sequence().back().i,node->get_sequence().back().j);
	int win_sum = 0;
	int simu_sum = 0;
	int simu[THREAD_NUM];
	int wins[THREAD_NUM];
	#pragma omp parallel for num_threads(THREAD_NUM)
	for (int i = 0; i < THREAD_NUM; i++) {
		simu[i] = 0;
		wins[i] = 0;
	}
	#pragma omp parallel for num_threads(THREAD_NUM)
	for (int i = 0; i < MAX_TRIAL; i++) {
		bool timeout = false;
		GoBoard* cur_board = new GoBoard(*board);
		int step = 0;
		while (step < 300) {
			step ++;
			Point nxt_move = generateRndNxtMove(cur_board);
			if (nxt_move.i < 0) {
				break;
			}

			cur_board->update_board(nxt_move, points);
			if (checkAbort()) {
				timeout = true;
				break;
			}
		}
		if (!timeout) {
			// std::cout << "Time :" <<  1000.0 * double(clock() - start) / CLOCKS_PER_SEC << std::endl;
			int score = cur_board->score();
			if ((score > 0 && cur_player == BLACK)
			        || (score < 0 && cur_player == WHITE)) {
				wins[omp_get_thread_num()]++;
			}
			// if (checkAbort()) break;
			simu[omp_get_thread_num()] ++;
		}
		// std::cout<<"Board size at end of game(in bytes):"<<cur_board->getMySize()<<std::endl;
		delete cur_board;
	}

	#pragma omp parallel for reduction(+:win_sum)
	for (int i = 0; i < THREAD_NUM; i++) {
		win_sum += wins[i];
	}
	#pragma omp parallel for reduction(+:simu_sum)
	for (int i = 0; i < THREAD_NUM; i++) {
		simu_sum += simu[i];
	}
	totalSimu += simu_sum;
	std::cout << "run_simulation end with wins/simu:" << win_sum << "/" << simu_sum << std::endl;
	delete board;
	return win_sum;
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
	std::vector<Point> moves_vec = generateAllMoves(cur_board);
	while (moves_vec.size() > 0) {
		Point nxt_move = moves_vec.back();
		moves_vec.pop_back();
		node->add_children(new TreeNode(node->get_sequence(), nxt_move));
	}
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
		struct timeval end;
		gettimeofday(&end, NULL);
		double delta = (end.tv_sec - startTime.tv_sec) * 1000000u + end.tv_usec - startTime.tv_usec;
		// std::cout<<"Cur time:"<<(std::clock() - startTime)/CLOCKS_PER_SEC<<std::endl;
		abort = delta > maxTime * 1000;
	}
	return abort;
}

Point Mcts::generateRndNxtMove(GoBoard* cur_board) {
	Point* point = createPoints(bd_size);
	std::vector<Point> moves_vec = cur_board->get_next_moves(point);
	int len = moves_vec.size();
	if (len == 0) {
		return Point(-1, -1);
	}
	srand (time(NULL));
	int idx = rand() % len;
	return moves_vec[idx];
}

std::vector<Point> Mcts::generateAllMoves(GoBoard* cur_board) {
	//std::cout<<cur_board.m_size<<std::endl;

	Point* point = createPoints(bd_size);
	std::vector<Point> moves_vec = cur_board->get_next_moves(point);
	int len = moves_vec.size();

	/* NOTE: point has not been freed yet !!!!!*/

	srand (time(NULL));
	for(int i = 0; i < len; i++){
		//	std::cout<<"swap"<<std::endl;
		int swapIndex = rand() % len;
		Point t = moves_vec[i];
		moves_vec[i] = moves_vec[swapIndex];
		moves_vec[swapIndex] = t;
	}
	return moves_vec;
}
Point* createPoints(int bd_size) {
	int len = bd_size + 2;
	Point* point = static_cast<Point*> (malloc(sizeof(Point) * len * len));
	for (int i = 0; i < len; i++) {
		for (int j = 0; j < len; j++) {
			point[i * len + j] = Point(i, j);
		}
	}
	return point;
}
GoBoard* Mcts::get_board(std::vector<Point> sequence) {
	Point* point = createPoints(bd_size);
	GoBoard* bd = new GoBoard(bd_size);
	for (std::vector<Point>::iterator it = sequence.begin(); it != sequence.end(); it++) {
		bd->update_board(*it, point);
	}
	return bd;
}