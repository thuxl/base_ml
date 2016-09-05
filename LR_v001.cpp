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
	N_attribute = 2;
	//signle_data = NULL;
}
LR::~LR()
{
	//release
	for (int i=0; i<N_train_data; i++ )
		delete [] data_x_set[i];
	data_x_set.clear();
	y_set.clear();
}



void LR::save_model()
{ 
    cout << "Saving model..." << endl;
    ofstream fout(model_param_filename );
	for (int i=0; i<N_attribute+1; i++)
		fout<<weight[i]<<"\t";
	fout << "\n";
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


void LR::read_parameters(int argc, char* argv[] )
{
	int i=1;
	for ( ; (i<argc) && (argv[i])[0]=='-'; i++) 
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
                N_attribute = atoi(argv[++i]);
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

void LR::read_samp_file() {
	//the format is m lines data each of which contains d+1 colomns (d attributes and 1 result)
    ifstream fin(sample_data_filename);
    if(!fin) {
        cerr << "Error opening file: " << sample_data_filename << endl;
        exit(0);
    }

	N_train_data = 0;
    string line_str;
    while (getline(fin, line_str))
    {
		double *_x = new double[N_attribute + 1]();
		char *p, *endptr, line[100]={0};
		
		int i=0, y=0;
		strcpy(line, line_str.c_str());
		p = strtok(line, "\t");
		while (p)
		{
			//cout<<i<<","<<p<<",";

			if ( i == N_attribute ){
        		y = atoi(p);
				_x[i] = 1.0;
			}
			else
				_x[i] = strtod(p, &endptr);
			i++;
			p = strtok(NULL, "\t");
		}
		//cout<<"x1="<<_x[0]<<", x2="<<_x[1]<<", x3="<<_x[2]<<", y="<<y<<", line end.\n";
		data_x_set.push_back(_x);
		y_set.push_back(y);
		N_train_data++;
    }
    fin.close();
}

double LR::wx_b(int data_i)
{//calculate w^T * x + b for data_i data in data_x_set
	double wx=0;  int i=0;
	for (; i<N_attribute; i++)
		wx += weight[i] * data_x_set[data_i][i];
	return wx + weight[i];
}
double LR::p1(int data_i)
{//p1_i = p(y=1|xi) = exp(wx_b(i)) / (1+exp(wx_b(i))) ;
	return exp(wx_b(data_i)) / (1.0 + exp(wx_b(data_i))); 
}
double LR::inner_product(int data_i)
{
	double s=0;
	for (int i=0; i<N_attribute + 1; i++)
		s += data_x_set[data_i][i] * data_x_set[data_i][i];
	return s;
}

void LR::train_model()
{
	//initialize w
	double w = (double) 1.0 / N_attribute;
	weight.clear();
	for (int i=0; i<N_attribute; i++)
		weight.push_back(w);
	weight.push_back(1.0); //b is initialized as 1.0

	//training by Newton method.
	//beta = (w; b), is a vector that is target to be approached.
	//d1_beta is first-order derivative  of l(beta) on beta.
	//d2_beta is second-order derivative of l(beta) on beta.
	vector <double> beta, d1_beta, d2_beta;
	for (int i=0; i<N_attribute + 1; i++){
		beta.push_back(weight[i]);
		d1_beta.push_back(0);
		d2_beta.push_back(0);
	}
	for (int k=0; k<N_max_loop; k++ )
	{
		for (int bi=0; bi<N_attribute+1; bi++){ //for each element in vector.
			//for d1_beta[bi] and d2_beta[bi]
			d1_beta[bi] = 0;
			d2_beta[bi] = 0;
			for (int i=0; i<N_train_data; i++ ){
				double pp1 = p1(i);
				d1_beta[bi] += data_x_set[i][bi] * (y_set[i] - pp1);
				d2_beta[bi] += inner_product(i) * pp1 * (1.0-pp1);
			}
			d1_beta[bi] = -d1_beta[bi]; 

			//for beta[bi]
			beta[bi] = beta[bi] - 1.0/d2_beta[bi] * d1_beta[bi]; 
		}
	}

	for (int i=0; i<N_attribute+1; i++)
		weight[i] = beta[i];
}

int main(int argc, char* argv[])
{

	//for (int i=0; i<argc; i++)
		//cout << "argv["<<i<<"]="<<argv[i] << ".\n";
	LR lr;
	lr.read_parameters(argc, argv);
	lr.read_samp_file();
	lr.train_model();
	lr.save_model();
	//lr.classify();

}
