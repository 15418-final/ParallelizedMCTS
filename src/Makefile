all: main

CXXFLAGS+=-g -Wall -I./utils -I./mcts -I./go -I./test -std=c++11
main : ./go/go.o ./mcts/mcts.o ./test/test

clean:
	rm -f ./go/*.o ./mcts/*.o ./test/*.o
	rm -f ./test/test -r 
	rm -r ./test/*.dSYM