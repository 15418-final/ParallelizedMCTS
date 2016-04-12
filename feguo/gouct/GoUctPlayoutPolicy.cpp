//----------------------------------------------------------------------------
/** @file GoUctPlayoutPolicy.cpp */
//----------------------------------------------------------------------------

#include "SgSystem.h"
#include "GoUctPlayoutPolicy.h"

#include <algorithm>
#include <boost/io/ios_state.hpp>

//----------------------------------------------------------------------------

GoUctPlayoutPolicyParam::GoUctPlayoutPolicyParam()
    : m_statisticsEnabled(false),
      m_useNakadeHeuristic(false),
      m_usePatternsInPlayout(true),
      m_usePatternsInPriorKnowledge(true),
      m_fillboardTries(0),
      m_patternGammaThreshold(50.f),
      m_knowledgeType(KNOWLEDGE_GREENPEEP),
      m_combinationType(COMBINE_MULTIPLY)
{ }

//----------------------------------------------------------------------------

const char* GoUctPlayoutPolicyTypeStr(GoUctPlayoutPolicyType type)
{
    BOOST_STATIC_ASSERT(_GOUCT_NU_DEFAULT_PLAYOUT_TYPE == 13);
    switch (type)
    {
    case GOUCT_FILLBOARD:
        return "Fillboard";
    case GOUCT_NAKADE:
        return "Nakade";
    case GOUCT_ATARI_CAPTURE:
        return "AtariCapture";
    case GOUCT_ATARI_DEFEND:
        return "AtariDefend";
    case GOUCT_LOWLIB:
        return "LowLib";
    case GOUCT_PATTERN:
        return "Pattern";
    case GOUCT_GAMMA_PATTERN:
    	return "GammaPattern";
    case GOUCT_REPLACE_CAPTURE:
        return "ReplaceCapture";
    case GOUCT_CAPTURE:
        return "Capture";
    case GOUCT_RANDOM:
        return "Random";
    case GOUCT_SELFATARI_CORRECTION:
        return "SelfAtariCorr";
    case GOUCT_CLUMP_CORRECTION:
        return "ClumpCorr";
    case GOUCT_PASS:
        return "Pass";
    default:
        return "?";
    }
}

//----------------------------------------------------------------------------

void GoUctPlayoutPolicyStat::Clear()
{
    m_nuMoves = 0;
    m_nonRandLen.Clear();
    m_moveListLen.Clear();
    std::fill(m_nuMoveType.begin(), m_nuMoveType.end(), 0);
}

void GoUctPlayoutPolicyStat::Write(std::ostream& out) const
{
    boost::io::ios_all_saver saver(out);
    out << std::fixed << std::setprecision(2)
        << SgWriteLabel("NuMoves") << m_nuMoves << '\n';
    for (int i = 0; i < _GOUCT_NU_DEFAULT_PLAYOUT_TYPE; ++i)
    {
        GoUctPlayoutPolicyType type = static_cast<GoUctPlayoutPolicyType>(i);
        size_t n = m_nuMoveType[type];
        out << SgWriteLabel(GoUctPlayoutPolicyTypeStr(type))
            << (m_nuMoves > 0 ? n * 100.0 / m_nuMoves : 0) << "%\n";
    }
    out << SgWriteLabel("NonRandLen");
    m_nonRandLen.Write(out);
    out << '\n'
        << SgWriteLabel("MoveListLen");
    m_moveListLen.Write(out);
    out << '\n';
}

//----------------------------------------------------------------------------
