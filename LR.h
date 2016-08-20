/********************************************************************
* Logistic Regression Classifier V0.10
* Implemented by Rui Xia(rxia@nlpr.ia.ac.cn) , Wang Tao（wangbogong@gmail.com）
* Last updated on 2012-6-12. 
*********************************************************************/
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

# define VERSION       "V0.10"
# define VERSION_DATE  "2012-6-12"

using namespace std;

const long  double min_threshold=1e-300;
/*
sparse_feat结构体表示的稀疏的存在结构，id_vec用于存放特征的id号，而value_vec用于存在对应的特征值。LR类是logistic regression 实现类，主要的两个函数train_online()与
classify_testing_file()，train_online()函数实现了logistic regression模型的随机梯度下降优化算法，classify_testing_file()函数实现了logistic regression模型对测试样本的预测。

*/
struct sparse_feat
{
    vector<int> id_vec;
    vector<float> value_vec;
};

class LR 
{
private:
    vector<sparse_feat> samp_feat_vec;
    vector<int> samp_class_vec;
    int feat_set_size;
    int class_set_size;
    vector< vector<float> > omega;
     
public:
    LR();
    ~LR();
    void save_model(string model_file);
    void load_model(string model_file);
    void load_training_file(string training_file);
    void init_omega();
    
    int train_online(int max_loop, double loss_thrd, float learn_rate, float lambda, int avg);
    vector<float> calc_score(sparse_feat &samp_feat);
    vector<float> score_to_prb(vector<float> &score);
    int score_to_class(vector<float> &score);
    
    float classify_testing_file(string testing_file, string output_file, int output_format);

private:
    void read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec);
    void update_online_ce(int samp_class, sparse_feat &samp_feat, float learn_rate, float lambda);
    void calc_loss_ce(double *loss, float *acc);
    float calc_acc(vector<int> &test_class_vec, vector<int> &pred_class_vec);    
    float sigmoid(float x);
    vector<string> string_split(string terms_str, string spliting_tag);

};
