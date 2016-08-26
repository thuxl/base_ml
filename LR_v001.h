//保证头文件只被编译一次
//#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <climits>
#include <math.h>
#include <time.h>

# define VERSION       "V0.01"
# define VERSION_DATE  "2016-8-26"

using namespace std;

struct watermelon
{
	double density;  //密度
	double sugar_content; //含糖率
	bool   good_t; //好瓜
};

class LR 
{
public:
	void read_parameters(int argc, char* argv[], char *training_filename, char* new_data_filename);
	int train();
	int save_model();
	int classify();
private:
	int N_train_data;
	int D_data;
};
