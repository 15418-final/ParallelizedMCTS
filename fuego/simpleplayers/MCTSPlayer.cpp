
#include "SgSystem.h"
#include "MCTSPlayer.h"

SgPoint MCTSPlayer::GenMove(const SgTimeRecord& time, SgBlackWhite toPlay) {
	SgPoint move = SG_NULLMOVE;
	double maxTime = timeControl.TimeForCurrentMove(time, false);

	// run mcts and get the best move
	Mcts* mcts = new Mcts(Board(), maxTime);
	move =  mcts->run();
	std::cout << SgPointUtil::PointToString(move) << std::endl;
	std::cout << "Board() size:" << Board().getMySize() << std::endl;
	size_t free_byte ;

	size_t total_byte ;

	cudaError cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

	if ( cudaSuccess != cuda_status ) {

		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

		exit(1);

	}
	delete mcts;
	return move;
}

