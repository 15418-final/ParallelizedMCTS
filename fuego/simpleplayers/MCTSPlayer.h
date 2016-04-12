#ifndef MCTS_PLAYER_H
#define MCTS_PLAYER_H

#include "SpSimplePlayer.h"
#include "SpMoveGenerator.h"
#include "../mcts/mcts.h"


//----------------------------------------------------------------------------

/** Plays moves by using MCTS */
class MCTSMoveGenerator
    : public Sp1PlyMoveGenerator
{
public:
    explicit MCTSMoveGenerator(const GoBoard& board)
        : Sp1PlyMoveGenerator(board)
    { }

    int Evaluate();
};

//----------------------------------------------------------------------------

/** Simple player using MCTSMoveGenerator */
class MCTSPlayer
    : public SpSimplePlayer
{
public:
    MCTSPlayer(const GoBoard& board)
        : SpSimplePlayer(board, new MCTSMoveGenerator(board))
    { }

    std::string Name() const
    {
        return "MCTS";
    }

    SgPoint GenMove(const SgTimeRecord& time, SgBlackWhite toPlay);

protected:
    bool UseFilter() const
    {
        return true;
    }
private:
    Mcts* mcts;
};

//----------------------------------------------------------------------------

#endif

