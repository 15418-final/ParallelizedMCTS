all: main

CXX=g++
CXXFLAGS+=-Wall -Iutils -Imcts -I./go -Itest -std=c++11 
LDLIBS+=-lstdc++
main : 
	$(CXX) -o run test/test.cpp go/go.cpp mcts/mcts.cpp $(CXXFLAGS)

clean:
	rm -f run