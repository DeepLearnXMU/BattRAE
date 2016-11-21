#include "BattRAE.h"

BattRAE::BattRAE(Parameter* para, Vocabulary* vocab){
	this -> para = para;
	this -> vocab = vocab;


	this -> save_times = 0;
	this -> best_model = -1;
	this -> best_score = -10000000;
	
	//{ allocate the memory
	__LBFGSALLOC__(x, get_x_size());
	Map<VectorLBFGS>(x, get_x_size()).setRandom();
	//}
	this -> loadnet();
}
BattRAE::~BattRAE(){ __LBFGSFREE__(x);}

int BattRAE::get_src_dim(){
	static int dim = atoi(para -> get_para("[src_dim]").c_str());
	return dim;
}
int BattRAE::get_tgt_dim(){
	static int dim = atoi(para -> get_para("[tgt_dim]").c_str());
	return dim;
}
int BattRAE::get_att_dim(){
	static int dim = atoi(para -> get_para("[att_dim]").c_str());
	return dim;
}
int BattRAE::get_sem_dim(){
	static int dim = atoi(para -> get_para("[sem_dim]").c_str());
	return dim;
}
int BattRAE::get_src_size(){
	static int dim = vocab -> get_source_size() * get_src_dim();
	return dim;
}
int BattRAE::get_tgt_size(){
	static int dim = vocab -> get_target_size() * get_tgt_dim();
	return dim;
}
int BattRAE::get_vocab_size(){
	static int dim = get_src_size() + get_tgt_size();
	return dim;
}
int BattRAE::get_src_rae_size(){
	int dim = get_src_dim();
	return dim * 2 * dim + dim + 2 * dim * dim + 2 * dim;
}
int BattRAE::get_tgt_rae_size(){
	int dim = get_tgt_dim();
	return dim * 2 * dim + dim + 2 * dim * dim + 2 * dim;
}
int BattRAE::get_att_size(){
	int src_dim = get_src_dim();
	int tgt_dim = get_tgt_dim();
	int att_dim = get_att_dim();

	return att_dim * src_dim + att_dim * tgt_dim + att_dim;
}
int BattRAE::get_sem_size(){
	int src_dim = get_src_dim();
	int tgt_dim = get_tgt_dim();
	int sem_dim = get_sem_dim();

	return sem_dim * src_dim + sem_dim * tgt_dim + sem_dim + sem_dim * sem_dim + 1;
}
int BattRAE::get_wb_size(){
	return get_src_rae_size()
		+ get_tgt_rae_size()
		+ get_att_size()
		+ get_sem_size()
	;
}
int BattRAE::get_x_size(){
	return get_vocab_size()						// vocabulary size
		+ get_wb_size()						// weight&bias size
	;
}

void BattRAE::initnet(){
	// default initialization of word embeddings
	Map<VectorLBFGS> m_src_words(x, get_src_size());
	m_src_words.setRandom();
	int src_dim = get_src_dim();
	for(size_t i = 0; i < (size_t) vocab -> get_source_size(); ++ i){
		m_src_words.segment(i * src_dim, src_dim) /= 
			m_src_words.segment(i * src_dim, src_dim).norm();
	}

	Map<VectorLBFGS> m_tgt_words(x + get_src_size(), get_tgt_size());
	m_tgt_words.setRandom();
	int tgt_dim = get_tgt_dim();
	for(size_t i = 0; i < (size_t) vocab -> get_target_size(); ++ i){
		m_tgt_words.segment(i * tgt_dim, tgt_dim) /=
			m_tgt_words.segment(i * tgt_dim, tgt_dim).norm();
	}

	{
		string init_src_net_file = this -> para -> get_para("[init_src_net]");
		ifstream is(init_src_net_file.c_str());
		if(!is){
			__FILE_MSG__(
				"failed to read source net:\t" << init_src_net_file << endl
			);
		}else{
			cout << "#Starting loading source net" << endl;
			//{ init word embedding with pre-trained word embedding :)
			string line; getline(is, line); vector<string> v_str = split_str(line, SPACE);
			int _dim = atoi(v_str[1].c_str());
			if(_dim != src_dim){
				__FILE_MSG__(
					"man, the word dimension of init net should be the same of configuration"
				);
				exit(1);
			}
			//}

			while(getline(is, line)){
				if("" == (line = strip_str(line))) continue;
	
				v_str = split_str(line, SPACE);
				string word = v_str[0];
	
				// src?
				if(vocab->word2id.find(SRC+word) != vocab->word2id.end()){
					long id = vocab -> get_id(word, true);
					for(size_t i = 1; i < v_str.size(); ++ i){
						x[id * src_dim + i - 1] = atof(v_str[i].c_str());
					}
				}
			}
			is.close();
		}
	}
	{
		string init_tgt_net_file = this -> para -> get_para("[init_tgt_net]");
		fstream is(init_tgt_net_file.c_str());
		if(!is){
			__FILE_MSG__(
				"failed to read target net:\t" << init_tgt_net_file << endl
			);
	
		}else{
			cout << "#Starting loading target net" << endl;
			//{ init word embedding with pre-trained word embedding :)
			string line; getline(is, line); vector<string> v_str = split_str(line, SPACE);
			int _dim = atoi(v_str[1].c_str());
			if(_dim != tgt_dim){
				__FILE_MSG__(
					"man, the word dimension of init net should be the same of configuration"
				);
				exit(1);
			}
			//}
	
			while(getline(is, line)){
				if("" == (line = strip_str(line))) continue;

				v_str = split_str(line, SPACE);
				string word = v_str[0];
	
				// tgt?
				if(vocab->word2id.find(TGT+word) != vocab->word2id.end()){
					long id = vocab -> get_id(word, false);
					for(size_t i = 1; i < v_str.size(); ++ i){
						x[get_src_size() + (id - vocab -> get_source_size()) * tgt_dim + i - 1]
							= atof(v_str[i].c_str());
					}
				}
			}
			is.close();
		}
	}

	initWB();
}
void BattRAE::initWB(){
	std::default_random_engine generator(time(NULL));
  	std::normal_distribution<double> distribution(0.0, 0.01);
	for(int i = get_vocab_size(); i < get_x_size(); ++ i){
		x[i] = distribution(generator);
	}

	{
		int src_dim = get_src_dim();
		__DEFINE_WBS__(x + get_vocab_size(), src_dim);
		m_B_1.setZero(); m_B_2.setZero();
	}
	{
		int tgt_dim = get_tgt_dim();
		__DEFINE_WBS__(x + get_vocab_size() + get_src_rae_size(), tgt_dim);
		m_B_1.setZero(); m_B_2.setZero();
	}
	{
		int src_dim = get_src_dim();
		int tgt_dim = get_tgt_dim();
		int att_dim = get_att_dim();
		int sem_dim = get_sem_dim();

		__DEFINE_ATT_WBS__(x + get_vocab_size() + get_src_rae_size() + get_tgt_rae_size(), 
				src_dim, tgt_dim, att_dim, sem_dim);
		m_Aw_b.setZero(); m_Sw_b.setZero(); m_Sb.setZero();
	}
}

void BattRAE::savenet(bool is_best, bool inc_times){
	if (inc_times){
		++ save_times;
	}
	string prefix = "";
	if (is_best) prefix = "best_";
	string save_vocab = prefix + para -> get_para("[save_net_vocab]");
	if (inc_times){
       		save_vocab += "_" + num2str(save_times);
	}

	int src_dim = get_src_dim();
	int tgt_dim = get_tgt_dim();
	int src_num = vocab -> get_source_size();
	
	ofstream os(save_vocab.c_str());

	//{ meta information
	os << vocab -> word2id.size() << SPACE << src_dim << SPACE << tgt_dim << endl;
	//}

	//{ each word representation
	Map<VectorLBFGS> m_words(x, get_vocab_size());
	for(long i = 0; i < (long)vocab -> id2word.size(); ++ i){
		string word = vocab -> get_word(i);
		os << word << SPACE;
		if(i < src_num){
			os << m_words.segment((i - 0) * src_dim, src_dim).transpose() << endl;
		}else{
			os << m_words.segment(src_num * src_dim + (i - src_num) * tgt_dim, tgt_dim).transpose() << endl;
		}
	}
	//}
	os.close();
	
	string save_net = prefix + para -> get_para("[save_net]");
        if(inc_times){
		save_net += "_" + num2str(save_times);
	}
	os.open(save_net.c_str());
	os << Map<VectorLBFGS>(x, get_x_size()).transpose() << endl;
	os.close();
}
void BattRAE::loadnet(string sfile){
	string save_net = sfile;
	if(sfile == "")
		save_net = para -> get_para("[save_net]");
	ifstream in(save_net.c_str());
	if(!in){
		__FILE_MSG__(
			"failed to read:\t" << "\"" << save_net << "\""
		);
		return ;
	}

	string line = "";
	stringstream ss;
	getline(in, line);
	ss << line;
	copy(
		istream_iterator<lbfgsfloatval_t>(ss), 
                       istream_iterator<lbfgsfloatval_t>(),
           		x
	);
	if(getline(in, line)){
		__FILE_MSG__(
			"bad file format:\t" << endl <<
			"file:\t" << "\"" << save_net << "\"" << endl <<
			"line:\t" << "\"" << line << "\""
		);
		exit(1);
	}
	ss.clear(), ss.str("");
	in.close();
}

// TODO: includ pretrain wrt RAE alone?
void BattRAE::train(){
	// convert training file
	cout << "#Converting the training file" << endl;
	trainset.clear();
	this -> file_lines = 
		vocab -> convert_train_file(
		para -> get_para("[train_file]"), trainset);
	cout << "#Finishing training file conversion" << endl;
	// TrainIng
	initnet();
	_train_lbfgs();
}
void BattRAE::_train_lbfgs(){
	// batching training
	int echotimes = atoi(para -> get_para("[iter_num]").c_str());
	// prepare the training parameter
	lbfgs_parameter_t lbfgs_para;
	lbfgs_parameter_init(&lbfgs_para);
	lbfgs_para.max_iterations = echotimes;

	lbfgsfloatval_t fx = 0.0;
        int ret = lbfgs(get_x_size(), x, &fx, _evaluate, _progress, this, &lbfgs_para);
	printf("L-BFGS optimization terminated with status code = %d\n", ret);
        printf(" fx = %f\n", fx);

	cout << "best model:\t" << best_model << endl;
	cout << "best score:\t" << best_score << endl;
}

lbfgsfloatval_t BattRAE::bilattional_semantic(
	int src_dim,				// source word & phrase & sentence dimension
	int tgt_dim,				// target word & phrase & sentence dimension
	int att_dim,				// the attentional space dimension
	int sem_dim,				// the semantic space dimension

	int src_word_num,			// the source side word number in the vocabulary
	int src_vocab_size,			// the source side vocabulary size
	int total_vocab_size,			// the total vocabulary size
	int src_rae_size,			// the source side rae parameter size
	int tgt_rae_size,			// the target side rae parameter size

	lbfgsfloatval_t* theta,			// the whole parameters
	lbfgsfloatval_t* grand,			// the whole gradients
	lbfgsfloatval_t alpha,			// the alpha for loss balance from RAE & SEM

	string src_instance,			// the source side instance
	string tgt_instance,			// the target side instance

	BiattSemValue* bsv			// a structure of the required statistics
	){
	// 0. We can split the parameters first
	lbfgsfloatval_t* theta_src_word = theta;
	lbfgsfloatval_t* theta_tgt_word = theta + src_vocab_size;
	lbfgsfloatval_t* theta_src_rae = theta + total_vocab_size;
	lbfgsfloatval_t* theta_tgt_rae = theta_src_rae + src_rae_size;
	lbfgsfloatval_t* theta_att_sem = theta_tgt_rae + tgt_rae_size;
	lbfgsfloatval_t* grand_src_word = grand;
	lbfgsfloatval_t* grand_tgt_word = grand + src_vocab_size;
	lbfgsfloatval_t* grand_src_rae = grand + total_vocab_size;
	lbfgsfloatval_t* grand_tgt_rae = grand_src_rae + src_rae_size;

	// 1. We need build the tree structure from source side and target side respectively
	Tree* src_tree = new Tree(src_instance
		, src_dim
		, 0
		, bsv -> is_tree
		, alpha
		, theta_src_word
		, theta_src_rae
		, grand_src_word
		, grand_src_rae);
	Tree* tgt_tree = new Tree(tgt_instance
		, tgt_dim
		, src_word_num
		, bsv -> is_tree
		, alpha
		, theta_tgt_word
		, theta_tgt_rae
		, grand_tgt_word
		, grand_tgt_rae);

	// 2. Construct the RAE representation matrix
	int src_node_num = src_tree -> nodes.size();
	int tgt_node_num = tgt_tree -> nodes.size();
	// 2.1 Allocate memory for BiattSemValue
	// The tree structure means the training processure
	if(bsv -> is_tree){
		__LBFGSALLOC__(bsv->theta_src_att_mat, src_node_num * att_dim);		// S * n
		__LBFGSALLOC__(bsv->theta_tgt_att_mat, tgt_node_num * att_dim);		// T * n
		__LBFGSALLOC__(bsv->grand_src_att_mat, src_node_num * att_dim);		// S * n
		__LBFGSALLOC__(bsv->grand_tgt_att_mat, tgt_node_num * att_dim);		// T * n

		__LBFGSALLOC__(bsv->grand_src_att_score, src_node_num * src_node_num);	// S * S
		__LBFGSALLOC__(bsv->grand_tgt_att_score, tgt_node_num * tgt_node_num); 	// T * T
		__LBFGSALLOC__(bsv->theta_src_att_score, src_node_num * 1);		// S * 1
		__LBFGSALLOC__(bsv->theta_tgt_att_score, tgt_node_num * 1);		// T * 1

		__LBFGSALLOC__(bsv->theta_src_rae_mat, src_node_num * src_dim);		// S * n1
		__LBFGSALLOC__(bsv->theta_tgt_rae_mat, tgt_node_num * tgt_dim);		// T * n2

		__LBFGSALLOC__(bsv->grand_src_sem_rep, sem_dim * 1);			// ns * 1
		__LBFGSALLOC__(bsv->grand_tgt_sem_rep, sem_dim * 1);			// nt * 1
		__LBFGSALLOC__(bsv->theta_src_sem_rep, sem_dim * 1);			// ns * 1
		__LBFGSALLOC__(bsv->theta_tgt_sem_rep, sem_dim * 1);			// nt * 1

		__LBFGSALLOC__(bsv->theta_src_inst_representation, src_dim * 1);	// n1 * 1
		__LBFGSALLOC__(bsv->theta_tgt_inst_representation, tgt_dim * 1);	// n2 * 1

		__LBFGSALLOC__(bsv->grand_v_score, 1);					// 1 * 1
		__LBFGSALLOC__(bsv->grand_biattention_matrix, src_node_num * tgt_node_num);
	}
	__LBFGSALLOC__(bsv->biattention_matrix, src_node_num * tgt_node_num);

	MatrixLBFGS src_rae_mat(src_node_num, src_dim);
	MatrixLBFGS tgt_rae_mat(tgt_node_num, tgt_dim);
	for(int i = 0; i < src_node_num; ++ i){
		src_rae_mat.row(i) = Map<VectorLBFGS>(src_tree->nodes[i]->v_vector, src_dim).transpose();
	}
	for(int j = 0; j < tgt_node_num; ++ j){
		tgt_rae_mat.row(j) = Map<VectorLBFGS>(tgt_tree->nodes[j]->v_vector, tgt_dim).transpose();
	}

	// 3. Transform the representation into the attentional space
	__DEFINE_ATT_WBS__(theta_att_sem, src_dim, tgt_dim, att_dim, sem_dim);
	// 3.1 w*x + b
	MatrixLBFGS src_att_mat = src_rae_mat * m_Aw_s.transpose() + m_Aw_b.transpose().colwise().replicate(src_node_num);
	MatrixLBFGS tgt_att_mat = tgt_rae_mat * m_Aw_t.transpose() + m_Aw_b.transpose().colwise().replicate(tgt_node_num);
	// 3.2 f(z) <= tanh
	// 3-for gradient
	// We need the `gradient` of the src_att_mat as well as that of the tgt_att_mat
	src_att_mat = (src_att_mat.array().exp() - (-1 * src_att_mat).array().exp()).array()
		/ (src_att_mat.array().exp() + (-1 * src_att_mat).array().exp()).array();
	tgt_att_mat = (tgt_att_mat.array().exp() - (-1 * tgt_att_mat).array().exp()).array()
		/ (tgt_att_mat.array().exp() + (-1 * tgt_att_mat).array().exp()).array();
	// 3.3 f'(z)
	if(bsv->is_tree){
		Map<MatrixLBFGS> grand_src_att_mat(bsv->grand_src_att_mat, src_node_num, att_dim);
		Map<MatrixLBFGS> grand_tgt_att_mat(bsv->grand_tgt_att_mat, tgt_node_num, att_dim);
		grand_src_att_mat = 1.0 - src_att_mat.array().square();
		grand_tgt_att_mat = 1.0 - tgt_att_mat.array().square();
	}

	// 4. calculate the attention matrix & bi-attentional 
	// 4.1 S x n * n * T
	MatrixLBFGS bi_att_mat = src_att_mat * tgt_att_mat.transpose();
	bi_att_mat = 1.0 / (1.0 + (-1.0 * bi_att_mat).array().exp());
	Map<MatrixLBFGS>(bsv->biattention_matrix, src_node_num, tgt_node_num) = bi_att_mat;
	// 4.2 S, T
	VectorLBFGS src_att_score = bi_att_mat.rowwise().sum();
	VectorLBFGS tgt_att_score = bi_att_mat.colwise().sum();
	src_att_score /= (1.0 * tgt_node_num);
	tgt_att_score /= (1.0 * src_node_num);
	// 4.3 SoftMax
	// 4-for gradient
	// We need calculate the `gradient` of softMax in this scoring
	// We also need preserve the `src&tgt_att_mat` for the attention derivation
	src_att_score = (src_att_score.array() - src_att_score.maxCoeff()).array().exp();
	src_att_score /= src_att_score.sum();
	tgt_att_score = (tgt_att_score.array() - tgt_att_score.maxCoeff()).array().exp();
	tgt_att_score /= tgt_att_score.sum();
	// 4.4 SoftMax'
	if(bsv->is_tree){
		MatrixLBFGS src_att_score_diag = src_att_score.asDiagonal() * 1.0;
		Map<MatrixLBFGS> grand_src_att_score(bsv->grand_src_att_score, src_node_num, src_node_num);
		grand_src_att_score = src_att_score_diag - 1.0 * (src_att_score * src_att_score.transpose());

		MatrixLBFGS tgt_att_score_diag = tgt_att_score.asDiagonal() * 1.0;
		Map<MatrixLBFGS> grand_tgt_att_score(bsv->grand_tgt_att_score, tgt_node_num, tgt_node_num);
		grand_tgt_att_score = tgt_att_score_diag - 1.0 * (tgt_att_score * tgt_att_score.transpose());

		Map<MatrixLBFGS> grand_biattention_matrix(bsv->grand_biattention_matrix, src_node_num, tgt_node_num);
		grand_biattention_matrix = bi_att_mat.array() * (1.0 - bi_att_mat.array());
	}

	// 5. Generate the attentioned source&target representation
	// 5-for gradient
	// We need the `src_att_score` and `tgt_att_score` for prop into the matrix
	// We also need the `src_rae_mat` and `tgt_rae_mat` for prop into the attention scorer layer
 	VectorLBFGS src_inst_representation = src_rae_mat.transpose() * src_att_score;
	VectorLBFGS tgt_inst_representation = tgt_rae_mat.transpose() * tgt_att_score;
	// 5.1 preserving representation
	if(bsv->is_tree){
		Map<VectorLBFGS> theta_src_inst_representation(bsv->theta_src_inst_representation, src_dim * 1);
		theta_src_inst_representation = src_inst_representation;
		Map<VectorLBFGS> theta_tgt_inst_representation(bsv->theta_tgt_inst_representation, tgt_dim * 1);
		theta_tgt_inst_representation = tgt_inst_representation;
	}

	// 6. Transform the representation into the semantic space
	// 6.1 w*x + b
	VectorLBFGS src_sem_rep = m_Sw_s * src_inst_representation + m_Sw_b;
	VectorLBFGS tgt_sem_rep = m_Sw_t * tgt_inst_representation + m_Sw_b;
	// 6.2 f(z) <= tanh
	// 6-for gradient
	// We need the `gradient` of src_sem_rep and tgt_sem_rep for backprop
	src_sem_rep = (src_sem_rep.array().exp() - (-1 * src_sem_rep).array().exp()).array()
		/ (src_sem_rep.array().exp() + (-1 * src_sem_rep).array().exp()).array();
	tgt_sem_rep = (tgt_sem_rep.array().exp() - (-1 * tgt_sem_rep).array().exp()).array()
		/ (tgt_sem_rep.array().exp() + (-1 * tgt_sem_rep).array().exp()).array();
	// 6.3 f'(z)
	if(bsv->is_tree){
		Map<VectorLBFGS> grand_src_sem_rep(bsv->grand_src_sem_rep, sem_dim);
 	        grand_src_sem_rep = 1.0 - src_sem_rep.array().square();
		Map<VectorLBFGS> grand_tgt_sem_rep(bsv->grand_tgt_sem_rep, sem_dim);
	       	grand_tgt_sem_rep = 1.0 - tgt_sem_rep.array().square();
	}

	// 7. Calculate the semantic score
	// 7-for gradient
	// We need both `src_sem_rep` and `tgt_sem_rep` to obtain the source side and target side gradient
	// 7.1 I considered for several times, now i decide to convert this score with the tanh translation
	VectorLBFGS v_score = src_sem_rep.transpose() * m_Sw * tgt_sem_rep + m_Sb;
	// only the training stage, we want to tanh the scoring
	// this is to control the scale of the learned score
	if(bsv->is_tree){
		v_score = (v_score.array().exp() - (-1 * v_score).array().exp()).array()
			/ (v_score.array().exp() + (-1 * v_score).array().exp()).array();
	}
	lbfgsfloatval_t score = v_score(0);
	if(bsv->is_tree){
		Map<VectorLBFGS> grand_v_score(bsv->grand_v_score, 1);
		grand_v_score = 1.0 - v_score.array().square();
	}

	// 8. Preserving the internal variable for gradient training
	bsv->semScore = score;
	bsv->src_tree = src_tree;
	bsv->tgt_tree = tgt_tree;
	if(bsv->is_tree){
		Map<MatrixLBFGS> theta_src_att_mat(bsv->theta_src_att_mat, src_node_num, att_dim);
		theta_src_att_mat = src_att_mat;
		Map<MatrixLBFGS> theta_tgt_att_mat(bsv->theta_tgt_att_mat, tgt_node_num, att_dim);
		theta_tgt_att_mat = tgt_att_mat;

		Map<VectorLBFGS> theta_src_att_score(bsv->theta_src_att_score, src_node_num);
		theta_src_att_score = src_att_score;
		Map<VectorLBFGS> theta_tgt_att_score(bsv->theta_tgt_att_score, tgt_node_num);
		theta_tgt_att_score = tgt_att_score;

		Map<MatrixLBFGS> theta_src_rae_mat(bsv->theta_src_rae_mat, src_node_num, src_dim);
		theta_src_rae_mat = src_rae_mat;
		Map<MatrixLBFGS> theta_tgt_rae_mat(bsv->theta_tgt_rae_mat, tgt_node_num, tgt_dim);
		theta_tgt_rae_mat = tgt_rae_mat;

		Map<VectorLBFGS> theta_src_sem_rep(bsv->theta_src_sem_rep, sem_dim);
		theta_src_sem_rep = src_sem_rep;
		Map<VectorLBFGS> theta_tgt_sem_rep(bsv->theta_tgt_sem_rep, sem_dim);
		theta_tgt_sem_rep = tgt_sem_rep;
	}

	return score;
}
void BattRAE::bilattional_semantic_backprop(
	int src_dim,				// source word & phrase & sentence dimension
	int tgt_dim,				// target word & phrase & sentence dimension
	int att_dim,				// the attentional space dimension
	int sem_dim,				// the semantic space dimension

	int src_word_num,			// the source side word number in the vocabulary
	int src_vocab_size,			// the source side vocabulary size
	int total_vocab_size,			// the total vocabulary size
	int src_rae_size,			// the source side rae parameter size
	int tgt_rae_size,			// the target side rae parameter size

	lbfgsfloatval_t* theta,			// the whole parameters
	lbfgsfloatval_t* grand,			// the whole gradients
	lbfgsfloatval_t flag, 			// positive or negative

	BiattSemValue* bsv			// bilingual attentional semantic values
	){
	// 0. We can split the parameters first
	lbfgsfloatval_t* theta_src_rae = theta + total_vocab_size;
	lbfgsfloatval_t* theta_tgt_rae = theta_src_rae + src_rae_size;
	lbfgsfloatval_t* theta_att_sem = theta_tgt_rae + tgt_rae_size;
	lbfgsfloatval_t* grand_src_rae = grand + total_vocab_size;
	lbfgsfloatval_t* grand_tgt_rae = grand_src_rae + src_rae_size;
	lbfgsfloatval_t* grand_att_sem = grand_tgt_rae + tgt_rae_size;

	int src_node_num = bsv->src_tree->nodes.size();
	int tgt_node_num = bsv->tgt_tree->nodes.size();

	// 1. We can extract the internal value inside the `bsv`
	Map<MatrixLBFGS> theta_src_att_mat(bsv->theta_src_att_mat, src_node_num, att_dim);
	Map<MatrixLBFGS> theta_tgt_att_mat(bsv->theta_tgt_att_mat, tgt_node_num, att_dim);
	Map<VectorLBFGS> theta_src_att_score(bsv->theta_src_att_score, src_node_num);
	Map<VectorLBFGS> theta_tgt_att_score(bsv->theta_tgt_att_score, tgt_node_num);
	Map<MatrixLBFGS> theta_src_rae_mat(bsv->theta_src_rae_mat, src_node_num, src_dim);
	Map<MatrixLBFGS> theta_tgt_rae_mat(bsv->theta_tgt_rae_mat, tgt_node_num, tgt_dim);
	Map<VectorLBFGS> theta_src_sem_rep(bsv->theta_src_sem_rep, sem_dim);
	Map<VectorLBFGS> theta_tgt_sem_rep(bsv->theta_tgt_sem_rep, sem_dim);
	Map<VectorLBFGS> theta_src_inst_representation(bsv->theta_src_inst_representation, src_dim);
	Map<VectorLBFGS> theta_tgt_inst_representation(bsv->theta_tgt_inst_representation, tgt_dim);

	Map<MatrixLBFGS> grand_src_att_mat(bsv->grand_src_att_mat, src_node_num, att_dim);
	Map<MatrixLBFGS> grand_tgt_att_mat(bsv->grand_tgt_att_mat, tgt_node_num, att_dim);
	Map<MatrixLBFGS> grand_src_att_score(bsv->grand_src_att_score, src_node_num, src_node_num);
	Map<MatrixLBFGS> grand_tgt_att_score(bsv->grand_tgt_att_score, tgt_node_num, tgt_node_num);
	Map<VectorLBFGS> grand_src_sem_rep(bsv->grand_src_sem_rep, sem_dim);
	Map<VectorLBFGS> grand_tgt_sem_rep(bsv->grand_tgt_sem_rep, sem_dim);
	Map<MatrixLBFGS> grand_biattention_matrix(bsv->grand_biattention_matrix, src_node_num, tgt_node_num);

	Map<VectorLBFGS> grand_v_score(bsv->grand_v_score, 1);

	// 2. backprop from the score into the semantic values
	__DEFINE_ATT_DWBS__(grand_att_sem, src_dim, tgt_dim, att_dim, sem_dim);
	__DEFINE_ATT_WBS__(theta_att_sem, src_dim, tgt_dim, att_dim, sem_dim);
	VectorLBFGS sem_err = grand_v_score * flag;
	m_DSb += sem_err;
	m_DSw += (theta_src_sem_rep * theta_tgt_sem_rep.transpose()) * sem_err(0);

	VectorLBFGS err_src_sem_rep = ((m_Sw * theta_tgt_sem_rep) * sem_err(0)).array() * grand_src_sem_rep.array();
	VectorLBFGS err_tgt_sem_rep = ((theta_src_sem_rep.transpose() * m_Sw).transpose() * sem_err(0)).array() * grand_tgt_sem_rep.array();

	// 3. backprop from the commen semantic space to the rae space 
	m_DSw_s += err_src_sem_rep * theta_src_inst_representation.transpose();
	m_DSw_t += err_tgt_sem_rep * theta_tgt_inst_representation.transpose();
	m_DSw_b += err_src_sem_rep + err_tgt_sem_rep;

	VectorLBFGS err_src_inst_representation = m_Sw_s.transpose() * err_src_sem_rep;
	VectorLBFGS err_tgt_inst_representation = m_Sw_t.transpose() * err_tgt_sem_rep;

	// 4. backprop from the rae space into the rae tree
	MatrixLBFGS err_src_rae_mat = theta_src_att_score * err_src_inst_representation.transpose();
	for(int i = 0; i < src_node_num; ++ i){
		Map<VectorLBFGS>(bsv->src_tree->nodes[i]->v_cerror, src_dim) += 
		Map<MatrixLBFGS>(bsv->src_tree->nodes[i]->v_deriva, src_dim, src_dim) * err_src_rae_mat.row(i).transpose();
	}
	MatrixLBFGS err_tgt_rae_mat = theta_tgt_att_score * err_tgt_inst_representation.transpose();
	for(int j = 0; j < tgt_node_num; ++ j){
		Map<VectorLBFGS>(bsv->tgt_tree->nodes[j]->v_cerror, tgt_dim) += 
		Map<MatrixLBFGS>(bsv->tgt_tree->nodes[j]->v_deriva, tgt_dim, tgt_dim) * err_tgt_rae_mat.row(j).transpose();
	}
	VectorLBFGS err_src_att_score = grand_src_att_score * (theta_src_rae_mat * err_src_inst_representation);
	VectorLBFGS err_tgt_att_score = grand_tgt_att_score * (theta_tgt_rae_mat * err_tgt_inst_representation);

	// 5. backprop from rae space into the attention space
	MatrixLBFGS err_src_att_score_mat = (err_src_att_score * 1.0 / tgt_node_num).rowwise().replicate(tgt_node_num);
	MatrixLBFGS err_tgt_att_score_mat = (err_tgt_att_score * 1.0 / src_node_num).transpose().colwise().replicate(src_node_num);
	MatrixLBFGS err_bi_att_mat = grand_biattention_matrix.array() * (err_src_att_score_mat + err_tgt_att_score_mat).array();

	MatrixLBFGS err_src_att_mat = (err_bi_att_mat * theta_tgt_att_mat).array() * grand_src_att_mat.array();
	MatrixLBFGS err_tgt_att_mat = (err_bi_att_mat.transpose() * theta_src_att_mat).array() * grand_tgt_att_mat.array();

	// 6. backprop from the attention space into the rae space
	m_DAw_s += err_src_att_mat.transpose() * theta_src_rae_mat;
	m_DAw_t += err_tgt_att_mat.transpose() * theta_tgt_rae_mat;
	m_DAw_b += err_src_att_mat.colwise().sum() + err_tgt_att_mat.colwise().sum();

	err_src_rae_mat = err_src_att_mat * m_Aw_s;
	for(int i = 0; i < src_node_num; ++ i){
		Map<VectorLBFGS>(bsv->src_tree->nodes[i]->v_cerror, src_dim) +=
		Map<MatrixLBFGS>(bsv->src_tree->nodes[i]->v_deriva, src_dim, src_dim) * err_src_rae_mat.row(i).transpose();
	}
	err_tgt_rae_mat = err_tgt_att_mat * m_Aw_t;
	for(int j = 0; j < tgt_node_num; ++ j){
		Map<VectorLBFGS>(bsv->tgt_tree->nodes[j]->v_cerror, tgt_dim) +=
		Map<MatrixLBFGS>(bsv->tgt_tree->nodes[j]->v_deriva, tgt_dim, tgt_dim) * err_tgt_rae_mat.row(j).transpose();
	}

	// 7. backprop from the rae space into the words
	bsv->src_tree->backprop();
	bsv->tgt_tree->backprop();
}

void BattRAE::train_a_instance(string instance, 
	int src_dim,
	int tgt_dim,
	int att_dim,
	int sem_dim,
	int total_vocab_size,
	int src_vocab_size,
	int tgt_vocab_size,
	int src_rae_size,
	int tgt_rae_size,
	int src_word_num,
	int tgt_word_num,

	lbfgsfloatval_t alpha,
	lbfgsfloatval_t& error,
	long& ins_num,
	lbfgsfloatval_t margin,

	lbfgsfloatval_t& correct_sem_score,
	lbfgsfloatval_t& incorrect_source_score,
	lbfgsfloatval_t& incorrect_target_score,

	lbfgsfloatval_t* theta,
	lbfgsfloatval_t* grand){
	vector<string> v_str = split_str(instance, MOSES_SEP);
	if(v_str.size() != 4){
		__FILE_MSG__(
			"bad file format:\t" << "\"" << instance << "\""
		);
		exit(1);
	}

	string correct_source = v_str[0];			// the bilingual source phrase
	string correct_target = v_str[1];			// the bilingual target phrase
	string incorrect_source = v_str[2];			// the negative source phrase
	string incorrect_target = v_str[3];			// the negative target phrase

	// correct source
	Tree* src_tree = new Tree(correct_source
		, src_dim
		, 0
		, false
		, 0.100
		, theta
		, theta + total_vocab_size
		, grand
		, grand);
	correct_source = src_tree -> print_tree();
	delete src_tree;
	// correct target
	Tree* tgt_tree = new Tree(correct_target
		, tgt_dim
		, src_word_num
		, false
		, 0.100
		, theta + src_vocab_size
		, theta + total_vocab_size + src_rae_size
		, grand
		, grand);
	correct_target = tgt_tree -> print_tree();
	delete tgt_tree;
	// incorrect source
	incorrect_source = replace_word(correct_source, incorrect_source);
	// incorrect target
	incorrect_target = replace_word(correct_target, incorrect_target);

	// correct source Vs. correct target phrases
	BiattSemValue* correct_bsv = new BiattSemValue(true);
	bilattional_semantic(
		src_dim,
		tgt_dim,
		att_dim,
		sem_dim,

		src_word_num,
		src_vocab_size,
		total_vocab_size,
		src_rae_size,
		tgt_rae_size,

		theta,
		grand,
		alpha,

		correct_source,
		correct_target,

		correct_bsv
	);
	lbfgsfloatval_t scorer = correct_bsv->semScore;
	// correct source Vs. incorrect target phrase
	BiattSemValue* incorrect_target_bsv = new BiattSemValue(true);
	bilattional_semantic(
		src_dim,
		tgt_dim,
		att_dim,
		sem_dim,

		src_word_num,
		src_vocab_size,
		total_vocab_size,
		src_rae_size,
		tgt_rae_size,

		theta,
		grand,
		alpha,

		correct_source,
		incorrect_target,

		incorrect_target_bsv
	);
	// incorrect source Vs. correct target phrase
	BiattSemValue* incorrect_source_bsv = new BiattSemValue(true);
	bilattional_semantic(
		src_dim,
		tgt_dim,
		att_dim,
		sem_dim,

		src_word_num,
		src_vocab_size,
		total_vocab_size,
		src_rae_size,
		tgt_rae_size,

		theta,
		grand,
		alpha,

		incorrect_source,
		correct_target,

		incorrect_source_bsv
	);

	// max-margin training
	// we want maximium the correcting score, but minimizing the incorrect score
	// We need minimize the target objective, so the objective should be
	// 		max{0, 1 - correct + incorrect}
	// Thus, minimize upside, minimize incorrect, while maximize correct!
	// 1.0 correct_bsv Vs incorrect_target_bsv
	lbfgsfloatval_t _error = 0.0;
	_error = margin - correct_bsv->semScore + incorrect_target_bsv->semScore;
	if(_error > 0){
		error += _error * (1.0 - alpha);
		error += correct_bsv->src_tree->rfValue;
		error += correct_bsv->tgt_tree->rfValue;
		for(size_t i = 0; i < incorrect_target_bsv->src_tree->nodes.size(); ++i){
			const Node* cur_node = incorrect_target_bsv->src_tree->nodes[i];
			Map<VectorLBFGS>(cur_node->v_perror, cur_node->dim*2).setZero();
			Map<VectorLBFGS>(cur_node->v_oerror, cur_node->dim*2).setZero();
		}
		for(size_t i = 0; i < incorrect_target_bsv->tgt_tree->nodes.size(); ++i){
			const Node* cur_node = incorrect_target_bsv->tgt_tree->nodes[i];
			Map<VectorLBFGS>(cur_node->v_perror, cur_node->dim*2).setZero();
			Map<VectorLBFGS>(cur_node->v_oerror, cur_node->dim*2).setZero();
		}

		// correct ones
		bilattional_semantic_backprop(
			src_dim,
			tgt_dim,
			att_dim,
			sem_dim,

			src_word_num,
			src_vocab_size,
			total_vocab_size,
			src_rae_size,
			tgt_rae_size,

			theta,
			grand,
			-1.0 * (1.0 - alpha),
			correct_bsv
		);
		// incorrect target ones
		bilattional_semantic_backprop(
			src_dim,
			tgt_dim,
			att_dim,
			sem_dim,

			src_word_num,
			src_vocab_size,
			total_vocab_size,
			src_rae_size,
			tgt_rae_size,

			theta,
			grand,
			1.0 * (1.0 - alpha),
			incorrect_target_bsv
		);
	}
	// correct source Vs. correct target phrases
	// We need calcute this again, because the gradient is dirty upside
	if(correct_bsv != NULL) delete correct_bsv;
	correct_bsv = new BiattSemValue(true);
	bilattional_semantic(
		src_dim,
		tgt_dim,
		att_dim,
		sem_dim,

		src_word_num,
		src_vocab_size,
		total_vocab_size,
		src_rae_size,
		tgt_rae_size,

		theta,
		grand,
		alpha,

		correct_source,
		correct_target,

		correct_bsv
	);
	if(fabs(scorer - correct_bsv->semScore) > 1e-6) {
		__FILE_MSG__(
			"Two computation is inconsistent:\t" << "\"" << scorer << "\" Vs. \"" << correct_bsv->semScore << "\""
		);
		exit(1);
	}
	// 2.0 correct_bsv Vs incorrect_source_bsv
	_error = margin - correct_bsv->semScore + incorrect_source_bsv->semScore;
	if(_error > 0){
		error += _error * (1.0 - alpha);
		error += correct_bsv->src_tree->rfValue;
		error += correct_bsv->tgt_tree->rfValue;
		for(size_t i = 0; i < incorrect_source_bsv->src_tree->nodes.size(); ++i){
			const Node* cur_node = incorrect_source_bsv->src_tree->nodes[i];
			Map<VectorLBFGS>(cur_node->v_perror, cur_node->dim*2).setZero();
			Map<VectorLBFGS>(cur_node->v_oerror, cur_node->dim*2).setZero();
		}
		for(size_t i = 0; i < incorrect_source_bsv->tgt_tree->nodes.size(); ++i){
			const Node* cur_node = incorrect_source_bsv->tgt_tree->nodes[i];
			Map<VectorLBFGS>(cur_node->v_perror, cur_node->dim*2).setZero();
			Map<VectorLBFGS>(cur_node->v_oerror, cur_node->dim*2).setZero();
		}

		// correct ones
		bilattional_semantic_backprop(
			src_dim,
			tgt_dim,
			att_dim,
			sem_dim,

			src_word_num,
			src_vocab_size,
			total_vocab_size,
			src_rae_size,
			tgt_rae_size,

			theta,
			grand,
			-1.0 * (1.0 - alpha),
			correct_bsv
		);
		// incorrect target ones
		bilattional_semantic_backprop(
			src_dim,
			tgt_dim,
			att_dim,
			sem_dim,

			src_word_num,
			src_vocab_size,
			total_vocab_size,
			src_rae_size,
			tgt_rae_size,

			theta,
			grand,
			1.0 * (1.0 - alpha),
			incorrect_source_bsv
		);
	}

	// collect the semantic score
	correct_sem_score += correct_bsv->semScore;
	incorrect_source_score += incorrect_source_bsv->semScore;
	incorrect_target_score += incorrect_target_bsv->semScore;
	ins_num += 1;
	// free the memory
	if(correct_bsv != NULL) delete correct_bsv;
	if(incorrect_source_bsv != NULL) delete incorrect_source_bsv;
	if(incorrect_target_bsv != NULL) delete incorrect_target_bsv;
}

lbfgsfloatval_t BattRAE::test_a_instance(string src, string tgt, lbfgsfloatval_t* x){
	lbfgsfloatval_t* gTmp = NULL;
	__LBFGSALLOC__(gTmp, get_x_size());

	BiattSemValue* bsv = new BiattSemValue(false);
	lbfgsfloatval_t score = bilattional_semantic(
		get_src_dim(),
		get_tgt_dim(),
		get_att_dim(),
		get_sem_dim(),

		vocab->get_source_size(),
		get_src_size(),
		get_vocab_size(),
		get_src_rae_size(),
		get_tgt_rae_size(),

		x,
		gTmp,
		0.0,

		src,
		tgt,

		bsv
	);
	__LBFGSFREE__(gTmp);
	if(bsv != NULL) delete bsv;

	return score;
}

void BattRAE::test(){
	string tst_file = para->get_para("[test_file]");
	ifstream in(tst_file.c_str());
	if(!in){
		__FILE_MSG__(
			"failed to read:\t" << "\"" << tst_file << "\""
		);
		exit(1);
	}

	vector<string> segs = split_str(tst_file, "/");
	ofstream os((segs[segs.size()-1] + ".battrae").c_str());

	string line = "";
	vector<string> v_str;

	cout << "#Starting procesing test file" << endl;
	long count = 0;
	while(getline(in, line)){
		if((line = strip_str(line)) == "") continue;

		v_str = split_str(line, MOSES_SEP);
		if(v_str.size() < 2){
			__FILE_MSG__(
				"bad file format" << "\t\"" << line << "\""
			);
			exit(1);
		}

		string src = v_str[0], tgt = v_str[1];
		string cvt_src = "", cvt_tgt = "";

		vector<string> v_substr = split_str(src, SPACE);
		for(size_t i = 0; i < v_substr.size(); ++ i){
			cvt_src += num2str(vocab->get_id(v_substr[i], true)) + SPACE;
		}
		cvt_src = strip_str(cvt_src);

		v_substr = split_str(tgt, SPACE);
		for(size_t j = 0; j < v_substr.size(); ++ j){
			cvt_tgt += num2str(vocab->get_id(v_substr[j], false)) + SPACE;
		}
		cvt_tgt = strip_str(cvt_tgt);

		lbfgsfloatval_t score = test_a_instance(cvt_src, cvt_tgt, x);

		os << line << " " << score << endl;

		++ count;
		if (count % 10000 == 0){
			cout << "Testing file proceed " << count << " instances" << endl;
		}
	}
	in.close();
	os.close();
}

lbfgsfloatval_t BattRAE::dev_tun(lbfgsfloatval_t* cX){
	string dev_file = para->get_para("[dev_file]");
	ifstream in(dev_file.c_str());
	if(!in){
		__FILE_MSG__(
			"failed to read:\t" << "\"" << dev_file << "\""
		);
		exit(1);
	}
	if(cX == NULL) cX = x;

	string line = "";
	vector<string> v_str;
	lbfgsfloatval_t total_score = 0.0;

	while(getline(in, line)){
		if((line = strip_str(line)) == "") continue;

		v_str = split_str(line, MOSES_SEP);
		if(v_str.size() < 2){
			__FILE_MSG__(
				"bad file format" << "\t\"" << line << "\""
			);
			exit(1);
		}

		string src = v_str[0], tgt = v_str[1];
		string cvt_src = "", cvt_tgt = "";

		vector<string> v_substr = split_str(src, SPACE);
		for(size_t i = 0; i < v_substr.size(); ++ i){
			cvt_src += num2str(vocab->get_id(v_substr[i], true)) + SPACE;
		}
		cvt_src = strip_str(cvt_src);

		v_substr = split_str(tgt, SPACE);
		for(size_t j = 0; j < v_substr.size(); ++ j){
			cvt_tgt += num2str(vocab->get_id(v_substr[j], false)) + SPACE;
		}
		cvt_tgt = strip_str(cvt_tgt);

		total_score += test_a_instance(cvt_src, cvt_tgt, cX);
	}
	return total_score;
}

lbfgsfloatval_t BattRAE::evaluate(
	const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
){
	lbfgsfloatval_t* cX = const_cast<lbfgsfloatval_t*>(x);			// const convertion

	lbfgsfloatval_t alpha = atof(para -> get_para("[alpha]").c_str());
	lbfgsfloatval_t lambda_Word = atof(para -> get_para("[lambda_Word]").c_str());
	lbfgsfloatval_t lambda_RAE = atof(para -> get_para("[lambda_RAE]").c_str());
	lbfgsfloatval_t lambda_Att = atof(para -> get_para("[lambda_Att]").c_str());
	lbfgsfloatval_t lambda_Sem = atof(para -> get_para("[lambda_Sem]").c_str());

	int src_dim = get_src_dim();
	int tgt_dim = get_tgt_dim();
	int att_dim = get_att_dim();
	int sem_dim = get_sem_dim();

	int total_vocab_size = get_vocab_size();
	int src_vocab_size = get_src_size();
	int tgt_vocab_size = get_tgt_size();

	int src_rae_size = get_src_rae_size();
	int tgt_rae_size = get_tgt_rae_size();

	int src_word_num = vocab->get_source_size();
	int tgt_word_num = vocab->get_target_size();

	lbfgsfloatval_t margin = atof(para -> get_para("[margin]").c_str());

	lbfgsfloatval_t fX = 0.0;						// The final objective value
	long total_ins_num = 0;							// The total instance number
	Map<VectorLBFGS> m_grand(g, get_x_size());
	m_grand.setZero();							// The gradient summer
	lbfgsfloatval_t total_correct_score = 0.0;				// The final correct semantic score
	lbfgsfloatval_t total_incorrect_source_score = 0.0;			// The final incorrect source semantic score
	lbfgsfloatval_t total_incorrect_target_score = 0.0;			// The final incorrect target semantic score
        for(size_t i = 0; i < trainset.size(); ++ i){

		train_a_instance(trainset[i], 

			src_dim,
			tgt_dim,
			att_dim,
			sem_dim,
			total_vocab_size,
			src_vocab_size,
			tgt_vocab_size,
			src_rae_size,
			tgt_rae_size,
			src_word_num,
			tgt_word_num,

			alpha,
			fX,
			total_ins_num,
			margin,

			total_correct_score,
			total_incorrect_source_score,
			total_incorrect_target_score,

			cX,
			g
		);

		if(total_ins_num % 10000 == 0){
			cout << "runing " << total_ins_num << " instances" << endl;
		}
	}

	//{ collect results
	if(total_ins_num != (int)trainset.size()){
		__FILE_MSG__(
			"error in train set size:\t" << total_ins_num << " vs " << trainset.size()
		);
		exit(1);
	}else{
		cout << "Training step: " << step << " Total correct score:\t" << total_correct_score << endl;
		cout << "Training step: " << step << " Total incorrect source score:\t" << total_incorrect_source_score << endl;
		cout << "Training step: " << step << " Total incorrect target score:\t" << total_incorrect_target_score << endl;
	}

	//{ objective value
	fX = fX * 1.0 / total_ins_num;	
	m_grand /= total_ins_num;
	//}
	//{ Objective
	//{ 1. vocabulary
	fX += Map<VectorLBFGS>(cX, get_vocab_size()).array().square().sum() * lambda_Word * 1.0 / 2;
	m_grand.segment(0, get_vocab_size()) += Map<VectorLBFGS>(cX, get_vocab_size()) * lambda_Word;  // lambda * 1/2 * |L|^2
	//}
	
	lbfgsfloatval_t* theta = cX + total_vocab_size;
	lbfgsfloatval_t* grand = g + total_vocab_size;
	{
		__DEFINE_WBS__(theta, src_dim);
		fX +=   m_W_1.array().square().sum() * lambda_RAE * 1.0 / 2
	      		+ m_W_2.array().square().sum() * lambda_RAE * 1.0 / 2
		;
		__DEFINE_DWBS__(grand, src_dim);
		m_Dw_1 += m_W_1 * lambda_RAE;
		m_Dw_2 += m_W_2 * lambda_RAE;
	}
	theta = theta + src_rae_size;
	grand = grand + src_rae_size;
	{
		__DEFINE_WBS__(theta, tgt_dim);
		fX +=   m_W_1.array().square().sum() * lambda_RAE * 1.0 / 2
	      		+ m_W_2.array().square().sum() * lambda_RAE * 1.0 / 2
		;
		__DEFINE_DWBS__(grand, tgt_dim);
		m_Dw_1 += m_W_1 * lambda_RAE;
		m_Dw_2 += m_W_2 * lambda_RAE;
	}
	theta = theta + tgt_rae_size;
	grand = grand + tgt_rae_size;
	{
		__DEFINE_ATT_WBS__(theta, src_dim, tgt_dim, att_dim, sem_dim);
		fX += 	m_Aw_s.array().square().sum() * lambda_Att * 1.0 / 2
			+ m_Aw_t.array().square().sum() * lambda_Att * 1.0 / 2
			+ m_Sw_s.array().square().sum() * lambda_Sem * 1.0 / 2
			+ m_Sw_t.array().square().sum() * lambda_Sem * 1.0 / 2
			+ m_Sw.array().square().sum() * lambda_Sem * 1.0 / 2
		;
		__DEFINE_ATT_DWBS__(grand, src_dim, tgt_dim, att_dim, sem_dim);
		m_DAw_s += m_Aw_s * lambda_Att;
		m_DAw_t += m_Aw_t * lambda_Att;
		m_DSw_s += m_Sw_s * lambda_Sem;
		m_DSw_t += m_Sw_t * lambda_Sem;
		m_DSw += m_Sw * lambda_Sem;
	}
	//}
	//}

	return fX;
}

lbfgsfloatval_t BattRAE::_evaluate(
	void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
){
	return reinterpret_cast<BattRAE*>(instance)->evaluate(x, g, n, step);
}
	 
int BattRAE::_progress(
	void *instance,                                                           
        const lbfgsfloatval_t *x,
	const lbfgsfloatval_t *g,
	const lbfgsfloatval_t fx,
	const lbfgsfloatval_t xnorm,
	const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step,
	int n,
	int k,
	int ls
){
	return reinterpret_cast<BattRAE*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
}
	 
int BattRAE::progress(
	const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
){
	// save the parameters and find the correct ones
	savenet(false, false);
	lbfgsfloatval_t score = dev_tun();
	if(score > best_score){
		this->best_score = score;
		this->best_model = save_times;
		savenet(true, false);
	}
	cout << "Iteration:\t" << k << " Score:\t" << score << endl;

	printf("Iteration %d:\n", k);
        printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
        printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
       	printf("\n");
        return 0;
}
