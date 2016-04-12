//----------------------------------------------------------------------------
/** @file GoTimeControl.cpp
    See GoTimeControl.h. */
//----------------------------------------------------------------------------

#include "SgSystem.h"
#include "GoTimeControl.h"

#include "GoBoard.h"

using std::max;

//----------------------------------------------------------------------------

GoTimeControl::GoTimeControl(const GoBoard& bd)
    : m_bd(bd),
      m_finalSpace(0.75f)
{ }

float GoTimeControl::FinalSpace() const
{
    return m_finalSpace;
}

void GoTimeControl::GetPositionInfo(SgBlackWhite& toPlay,
                                    int& movesPlayed,
                                    int& estimatedRemainingMoves)
{
    toPlay = m_bd.ToPlay();
    movesPlayed = m_bd.Occupied().Size() / 2;
    int finalNumEmpty =
        int(float(m_bd.AllPoints().Size()) * (1.f - m_finalSpace));
    estimatedRemainingMoves = max(m_bd.TotalNumEmpty() - finalNumEmpty, 
                                  3 * m_bd.Size());
    estimatedRemainingMoves /= 2;
}

void GoTimeControl::SetFinalSpace(float finalspace)
{
    // Should be in [0:1]
    SG_ASSERT(finalspace > -0.1);
    SG_ASSERT(finalspace < 1.1);
    m_finalSpace = finalspace;
}

//----------------------------------------------------------------------------
