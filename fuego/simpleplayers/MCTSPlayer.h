#ifndef MCTS_PLAYER_H
#define MCTS_PLAYER_H

#include <boost/scoped_ptr.hpp>
#include <vector>
#include "GoBoard.h"
#include "GoBoardRestorer.h"
#include "GoPlayer.h"
#include "GoTimeControl.h"
#include "SgArrayList.h"
#include "SgNbIterator.h"
#include "SgNode.h"
#include "SgPointArray.h"
#include "SgRestorer.h"
#include "SgMpiSynchronizer.h"
#include "SgTime.h"
#include "SgTimer.h"

#include "mcts.h"


class MCTSPlayer
    : public GoPlayer,
      public SgObjectWithDefaultTimeControl
{
public:
    MCTSPlayer(const GoBoard& board)
        : GoPlayer(board), timeControl(Board())
    { 
        
    }

    std::string Name() const
    {
        return "MCTS";
    }

    SgPoint GenMove(const SgTimeRecord& time, SgBlackWhite toPlay);

    SgDefaultTimeControl& TimeControl()
    {
        return timeControl;
    }

    const SgDefaultTimeControl& TimeControl() const
    {
        return timeControl;
    }

private:
    GoTimeControl timeControl;
};

//----------------------------------------------------------------------------

#endif

