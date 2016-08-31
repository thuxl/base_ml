/*
 * This program is referred to the logistic regression in <机器学习> by 周志华, Tsinghua University Press.
 * Realized by xl.
 */

#include "LR_v001.h"

LR::LR()
{
	memset(training_data_filename, 0, 200);
	memset(model_param_filename, 0, 200);
	N_max_loop = 200;
}
LR::~LR()
{}



void LR::save_model(string model_file)
{ 
    cout << "Saving model..." << endl;
    ofstream fout(model_file.c_str());
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
            fout << omega[k][j] << " ";
        }
        fout << endl;
    }
    fout.close();
}

void LR::print_help() 
{
    cout << "\nLR training module, " << VERSION << ", " << VERSION_DATE << "\n\n"
        << "usage: pLR_v001 [options] training_data_file model_parameter_file \n\n"
        << "options: -h        -> help\n"
        << "         -n int    -> maximal iteration loops (default 200)\n"
        //<< "         -m double -> minimal loss value decrease (default 1e-03)\n"
        //<< "         -r double -> regularization parameter lambda of gaussian prior (default 0)\n"        
        //<< "         -l float  -> learning rate (default 1.0)\n"
        //<< "         -a        -> 0: final weight (default)\n"
        //<< "                   -> 1: average weights of all iteration loops\n"
        //<< "         -u [0,1]  -> 0: initial training model (default)\n"
        //<< "                   -> 1: updating model (pre_model_file is needed)\n" 
        << endl;
}

void LR::read_parameters(int argc, char* argv[], char *training_filename, char* new_data_filename)
{
	for (int i = 1; (i<argc) && (argv[i])[0]=='-'; i++) 
    {
        switch ((argv[i])[1]) {
            case 'h':
                print_help();
                exit(0);
            case 'n':
                N_max_loop = atoi(argv[++i]);
                break;
			default:
                cout << "Unrecognized option: " << argv[i] << "!" << endl;
                print_help();
                exit(0);
        }
    }
    
    if ((i+1)>=argc) 
    {
        cout << "Not enough parameters!" << endl;
        print_help();
        exit(0);
    }

	strcpy (training_data_filename, argv[i]);
    strcpy (model_param_filename, argv[i+1]);


}

int main(int argc, char* argv[])
{

	for (int i=0; i<argc; i++)
	{
		cout << "argv["<<i<<"]="<<argv[i] << ".\n";
	}
}
