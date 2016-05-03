noinst_LIBRARIES = libfuego_simpleplayers.a

libfuego_simpleplayers_a_SOURCES = \
SpAveragePlayer.cpp \
SpCapturePlayer.cpp \
SpDumbTacticalPlayer.cpp \
SpGreedyPlayer.cpp \
SpInfluencePlayer.cpp \
SpLadderPlayer.cpp \
SpLibertyPlayer.cpp \
SpMaxEyePlayer.cpp \
SpMinLibPlayer.cpp \
SpMoveGenerator.cpp \
SpRandomPlayer.cpp \
SpSafePlayer.cpp \
SpSimplePlayer.cpp \
MCTSPlayer.cpp \
mcts_cuda.cu \
CudaGo.cu \
SpUtil.cpp

noinst_HEADERS = \
SpAveragePlayer.h \
SpCapturePlayer.h \
SpDumbTacticalPlayer.h \
SpGreedyPlayer.h \
SpInfluencePlayer.h \
SpLadderPlayer.h \
SpLibertyPlayer.h \
SpMaxEyePlayer.h \
SpMinLibPlayer.h \
SpMoveGenerator.h \
SpRandomPlayer.h \
SpSafePlayer.h \
SpSimplePlayer.h \
MCTSPlayer.h \
mcts.h \
CudaGo.h \
point.h \
deque.h \
SpUtil.h

AM_CXXFLAGS = -fopenmp

.cu.o:
    $(NVCC) $(NVCCFLAGS)  -c -o $@ $<

libfuego_simpleplayers_a_CPPFLAGS = \
$(BOOST_CPPFLAGS) \
-I@top_srcdir@/gtpengine \
-I@top_srcdir@/smartgame \
-I@top_srcdir@/go

DISTCLEANFILES = *~
