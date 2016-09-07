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
{//row * column = number
	double s=0;
	for (int i=0; i<N_attribute + 1; i++)
		s += data_x_set[data_i][i] * data_x_set[data_i][i];
	return s;
}

void LR::get_d1_beta(vector <double> &d1_beta )
{
	for (int bi=0; bi<N_attribute + 1; bi++){ //for each element in vector.
		d1_beta[bi] = 0;
		for (int i=0; i<N_train_data; i++ ){
			double pp1 = p1(i);
			d1_beta[bi] += data_x_set[i][bi] * (y_set[i] - pp1);
		}
		d1_beta[bi] = -d1_beta[bi];
	}

}

void LR::get_d2_beta(vector < vector<double> > &d2_beta)
{//column * row = matrix
	for (int i=0; i< N_attribute + 1; i++){
	for (int j=0; j< N_attribute + 1; j++)
	{
		d2_beta[i][j] = 0;
		for (int k=0; k<N_train_data; k++)
		{
			double pp1 = p1(k);
			d2_beta[i][j] += data_x_set[k][i] * data_x_set[k][j] * pp1 * (1-pp1);
		}
	}
	}
}

//计算每一行每一列的每个元素所对应的余子式，组成A*
void  LR::get_Algebraic_Cofactor(vector < vector<double> > A, vector < vector<double> > &ans, int n)
{
    if(n==1)
    {
        ans[0][0] = 1;
        return;
    }
    int i,j,k,t;
	vector < vector<double> > temp(n-1, vector<double> (n-1,0));
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            for(k=0;k<n-1;k++)
            {
                for(t=0;t<n-1;t++)
                {
                    temp[k][t] =  A[k>=i?k+1:k][t>=j?t+1:t] ;
                }
            }


            ans[j][i]  =  get_det(temp,n-1);
            if((i+j)%2 == 1)
            {
                ans[j][i] = - ans[j][i];
            }
        }
    }
}
//按第一行展开计算|A|
double LR::get_det(vector < vector<double> > A, int n)
{
    if(n==1)
    {
        return A[0][0];
    }
    double ans = 0;
    vector <vector <double> > temp (n-1, vector<double>(n-1, 0) );
    int i,j,k;
    for(i=0;i<n;i++)
    {//the i'th element in 0 row.
        for(j=0;j<n-1;j++)
        {
            for(k=0;k<n-1;k++)
                temp[j][k] = A[j+1][(k>=i)?k+1:k] ;
        }
        double det_n_1 = get_det(temp,n-1);
        if(i%2==0)
        {
            ans += A[0][i]*det_n_1 ;
        }
        else
        {
            ans -=  A[0][i]*det_n_1 ;
        }
    }
    return ans;
}
//得到给定矩阵src的逆矩阵保存到des中。
bool LR::GetMatrixInverse(vector < vector<double> > src,  vector < vector<double> > &des, int n)
{
    double det=get_det(src,n);
    vector <vector <double> > ac(n, (vector <double> (n,0)));
    if(det==0)
    {
        return false;
    }
    else
    {
        get_Algebraic_Cofactor(src,ac, n);
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                des[i][j]=ac[i][j]/det;
            }

        }
    }


    return true;

}

void LR::matrix_multiply_vector(vector < vector<double> > A,  vector < double > B, vector < double > &C, int m, int n )
{//A m*n multiplies B n*1, get C m*1
	for (int i=0; i<m; i++){
		C[i] = 0;
		for (int j=0; j<n; j++)
			C[i] += A[i][j] * B[j];
	}
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
	vector <double> beta(N_attribute + 1, 0), d1_beta(N_attribute + 1, 0) , d2invers_m_d1(N_attribute + 1, 0);
    vector <vector <double> > d2_beta(N_attribute + 1, (vector <double> (N_attribute + 1, 0)));
    vector <vector <double> > d2_beta_inv(N_attribute + 1, (vector <double> (N_attribute + 1, 0)));
	for (int i=0; i<N_attribute + 1; i++)
		beta[i]= weight[i];
	
	for (int k=0; k<N_max_loop; k++ )
	{
		cout<<"\nloop "<<k<<":\n";

		get_d1_beta( d1_beta );
		get_d2_beta( d2_beta );
		GetMatrixInverse( d2_beta ,  d2_beta_inv, N_attribute + 1);
		matrix_multiply_vector(d2_beta_inv, d1_beta, d2invers_m_d1, N_attribute + 1, N_attribute + 1);

		for (int bi=0; bi<N_attribute+1; bi++){ //for each element in vector.
			//for beta[bi]
			beta[bi] = beta[bi] - d2invers_m_d1[bi];
			weight[bi] = beta[bi]; //update weight

			cout<<"attribute ["<<bi<<"]: d1_beta="<<d1_beta[bi]<<", beta="<<beta[bi]<<";\n";
		}
	}

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
