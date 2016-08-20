all: pdecision_tree pLR_train pLR_classify
clean: rm p*

pdecision_tree: decision_tree.cpp
	g++ decision_tree.cpp -o pdecision_tree

pLR_train: LR.cpp  LR.h  LR_train.cpp
	g++ LR.cpp  LR.h  LR_train.cpp -o pLR_train

pLR_classify: LR.cpp  LR.h LR_classify.cpp 
	g++ LR.cpp  LR.h LR_classify.cpp -o pLR_classify
