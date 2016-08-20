#include <cstdlib>
#include <iostream>
#include <cstring>
#include "LR.h"

using namespace std;


void print_help()
{
    cout << "\nOpenPR-LR classification module, " << VERSION << ", " << VERSION_DATE << "\n\n"
        << "usage: LR_classify [options] testing_file model_file output_file\n\n"
        << "options: -h        -> help\n"
        << "         -f [0..2] -> 0: only output class label (default)\n"
        << "                   -> 1: output class label with log-likelihood (weighted sum)\n"
        << "                   -> 2: output class label with soft probability\n"
        << endl;
}

void read_parameters(int argc, char *argv[], char *testing_file, char *model_file, 
                        char *output_file, int *output_format) 
{
    // set default options
    *output_format = 0;
    int i;
    for (i = 1; (i<argc) && (argv[i])[0]=='-'; i++)
    {
        switch ((argv[i])[1]) {
            case 'h':
                print_help();
                exit(0);
            case 'f':
                *output_format = atoi(argv[++i]);
                break;
            default:
                cout << "Unrecognized option: " << argv[i] << "!" << endl;
                print_help();
                exit(0);
        }
    }
    
    if ((i+2)>=argc)
    {
        cout << "Not enough parameters!" << endl;
        print_help();
        exit(0);
    }
    strcpy(testing_file, argv[i]);
    strcpy(model_file, argv[i+1]);
    strcpy(output_file, argv[i+2]);
}

int LR_classify(int argc, char *argv[])
{
    char testing_file[200];
    char model_file[200];
    char output_file[200];
    int output_format;
    read_parameters(argc, argv, testing_file, model_file, output_file, &output_format);
    LR LR;
    LR.load_model(model_file);
    float acc = LR.classify_testing_file(testing_file, output_file, output_format);
    cout << "Accuracy: " << acc << endl;
    //ofstream outfile("d:\\result.txt",ios::app);
    //outfile<<testing_file<<"\t"<<acc<<endl;
    return 0;
}

int main(int argc, char *argv[])
{
    return LR_classify(argc, argv);
}

/*
5.usage

算法的处理样本的数据格式与libsvm样本数据格式一致，均采用稀疏的存储结构。

模型训练：> lr_train -n 50 -m 1e-06  data/train.samp result/ldf.mod。训练样本保存在data/train.samp文件，模型训练迭代的最大次数是50，最小损失值是1e-06，模型文件存储在result/ldf.mod文件。

分类预测：> lr_classify data/test.samp data/ldf.mod result/ldf.out。测试样本存放在data/test.samp文件，样本输出的结果文件存在在result/ldf.out文件。
*/
