all: pdecision_tree
clean: rm p*
pdecision_tree: decision_tree.cpp
	g++ decision_tree.cpp -o pdecision_tree
