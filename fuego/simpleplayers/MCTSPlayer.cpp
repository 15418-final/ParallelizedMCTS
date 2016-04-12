
#include "SgSystem.h"
#include "MCTSPlayer.h"


int MCTSMoveGenerator::Evaluate()
{
    // We are Opponent since this is after executing our move
    SgBlackWhite player = m_board.Opponent();
    
    return 0;
}

