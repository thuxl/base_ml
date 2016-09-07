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
#include <string.h>

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
	double wx_b(int data_i);
	double p1(int data_i);
	double inner_product(int data_i);
	void get_d1_beta(vector <double> &d1_beta );
	void get_d2_beta(vector < vector<double> > &d2_beta);
	void get_Algebraic_Cofactor(vector < vector<double> > A, vector < vector<double> > &ans, int n);
	double get_det(vector < vector<double> > A, int n);
	bool GetMatrixInverse(vector < vector<double> > src,  vector < vector<double> > &des, int n);
	void matrix_multiply_vector(vector < vector<double> > A,  vector < double > B, vector < double > &C, int m, int n );
	void train_model();
	void save_model();
	//int classify();
private:
	int N_train_data;
	int N_attribute;
    vector<double>  weight; //the last element of weight is b;
    
	vector<double *>  data_x_set;
	vector<int>       y_set;
	//double*  signle_data;
	

	int N_max_loop;
    char sample_data_filename[200];
    char model_param_filename[200];

};
