
#include "SgSystem.h"
#include "MCTSPlayer.h"

SgPoint MCTSPlayer::GenMove(const SgTimeRecord& time, SgBlackWhite toPlay) {
	SgPoint move = SG_NULLMOVE;
	double maxTime = timeControl.TimeForCurrentMove(time, false);

    // run mcts and get the best move
    Mcts* mcts = new Mcts(Board(), maxTime);
    move =  mcts->run();

 	delete mcts;
    return move;
}

