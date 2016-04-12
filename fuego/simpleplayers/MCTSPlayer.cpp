
#include "SgSystem.h"
#include "MCTSPlayer.h"

SgPoint MCTSPlayer::GenMove(const SgTimeRecord& time, SgBlackWhite toPlay) {
	SgPoint move = SG_NULLMOVE;
	double maxTime = timeControl.TimeForCurrentMove(time, false);

    // run mcts and get the best move
 /*
    mcts = new mcts(board, toPlay, maxTime);
    mcts.run();
    move = mcts.getBestMove();
 */
    return move;
}

