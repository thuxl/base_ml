all: pdecision_tree pLR_train pLR_classify pLR_v001
clean: rm p*

pdecision_tree: decision_tree.cpp
	g++ decision_tree.cpp -o pdecision_tree

pLR_train: LR.cpp  LR.h  LR_train.cpp
	g++ LR.cpp  LR.h  LR_train.cpp -o pLR_train

pLR_classify: LR.cpp  LR.h LR_classify.cpp 
	g++ LR.cpp  LR.h LR_classify.cpp -o pLR_classify

pLR_v001: LR_v001.cpp LR_v001.h
	g++ -g LR_v001.cpp LR_v001.h -o pLR_v001
