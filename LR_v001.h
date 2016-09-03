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
	bool   good_tag; //好瓜
};

class LR 
{
public:
	LR();
	~LR();
	void read_parameters(int argc, char* argv[]);
	void print_help();
	void read_samp_file();
	//int train();
	void save_model();
	//int classify();
private:
	int N_train_data;
	int N_data_attribute;
    vector<double>  weight;
	double    b;
    
	vector<double *>  data_x_set;
	vector<int>       y_set;
	//double*  signle_data;
	

	int N_max_loop;
    char sample_data_filename[200];
    char model_param_filename[200];

};
