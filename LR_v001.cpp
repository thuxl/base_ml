/*
 * This program is referred to the logistic regression in <机器学习> by 周志华, Tsinghua University Press.
 * Realized by xl.
 */

#include "LR_v001.h"

LR::LR()
{
	memset(sample_data_filename, 0, 200);
	memset(model_param_filename, 0, 200);
	N_max_loop = 200;
	N_train_data = 0;
	N_data_attribute = 2;
	//signle_data = NULL;
}
LR::~LR()
{
	//release
	for (int i=0; i<N_train_data; i++ )
		delete [] data_set[i];
	data_set.clear();
}



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
        << "         -l int    -> maximal iteration loops (default 200)\n"
        << "         -d int    -> the number of attributes of data (default 2)\n"
        //<< "         -m int    -> the number of training data (default 17)\n"
        //<< "         -m double -> minimal loss value decrease (default 1e-03)\n"
        //<< "         -r double -> regularization parameter lambda of gaussian prior (default 0)\n"
        //<< "         -l float  -> learning rate (default 1.0)\n"
        //<< "         -a        -> 0: final weight (default)\n"
        //<< "                   -> 1: average weights of all iteration loops\n"
        //<< "         -u [0,1]  -> 0: initial training model (default)\n"
        //<< "                   -> 1: updating model (pre_model_file is needed)\n" 
        << endl;
}

void LR::read_samp_file() {
	//the format is m lines data each of which contains d+1 colomns (d attributes and 1 result)
    ifstream fin(sample_data_filename);
    if(!fin) {
        cerr << "Error opening file: " << samp_file << endl;
        exit(0);
    }
	N_train_data = 0;
    string line_str;
    while (getline(fin, line_str))
    {
		char *endptr;
        size_t margin_pos = line_str.find_first_of("\t");
		double v = strtod(line_str.substr(0, margin_pos).c_str(), &endptr);
		string sub1 = line_str.substr(margin_pos+1);
        margin_pos = sub1.find_first_of("\t");
		double v2 = strtod(sub1.substr(0, margin_pos).c_str(), &endptr);
        margin_pos = sub1.find_last_of("\t");
        int y = atoi(line_str.substr(margin_pos+1, 1).c_str());



        samp_class_vec.push_back(class_id);
        string terms_str = line_str.substr(class_pos+1);
        sparse_feat samp_feat;
        samp_feat.id_vec.push_back(0); // bias
        samp_feat.value_vec.push_back(1); // bias
        if (terms_str != "") 
        {
            vector<string> fv_vec = string_split(terms_str, " ");
            for (vector<string>::iterator it = fv_vec.begin(); it != fv_vec.end(); it++) 
            {
                size_t feat_pos = it->find_first_of(":");
                int feat_id = atoi(it->substr(0, feat_pos).c_str());
                float feat_value = (float)atof(it->substr(feat_pos+1).c_str());
                samp_feat.id_vec.push_back(feat_id);
                samp_feat.value_vec.push_back(feat_value);
            }
        }
        samp_feat_vec.push_back(samp_feat);
    }
    fin.close();
}


void LR::read_parameters(int argc, char* argv[], char *training_filename, char* new_data_filename)
{
	for (int i = 1; (i<argc) && (argv[i])[0]=='-'; i++) 
    {
        switch ((argv[i])[1]) {
            case 'h':
                print_help();
                exit(0);
            case 'l':
                N_max_loop = atoi(argv[++i]);
                break;
            //case 'm':
                //N_train_data = atoi(argv[++i]);
                //break;
            case 'd':
                N_data_attribute = atoi(argv[++i]);
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

	strcpy (sample_data_filename, argv[i]);
    strcpy (model_param_filename, argv[i+1]);



}

int main(int argc, char* argv[])
{

	for (int i=0; i<argc; i++)
	{
		cout << "argv["<<i<<"]="<<argv[i] << ".\n";
	}
}
