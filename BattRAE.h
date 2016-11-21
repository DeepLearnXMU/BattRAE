#ifndef __BATTRAE_H__
#define __BATTRAE_H__

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>

#include <lbfgs.h>
#include <Eigen/Dense>

#include "Constant.h"
#include "Util.h"

#include "Node.h"
#include "Tree.h"
#include "Parameter.h"
#include "Vocabulary.h"
#include "BiattSemValue.h"

using namespace std;
using namespace Eigen;

static vector<string> trainset;

class BattRAE{
public:
	// init class
	BattRAE(Parameter* para, Vocabulary* vocab);
	~BattRAE();
	// training
	void train();
	// For evaluation
	void test();

private:
	// obtain important information from parameters
	int get_src_dim();
	int get_tgt_dim();
	int get_att_dim();
	int get_sem_dim();
	// compute important statistics
	int get_src_size();
	int get_tgt_size();
	int get_vocab_size();
	int get_src_rae_size();
	int get_tgt_rae_size();
	int get_att_size();
	int get_sem_size();
	int get_wb_size();
	int get_x_size();

	// initial the network
	void initnet();
	void initWB();

	// save the newtork
	void savenet(bool is_best=false, bool inc_times=true);
	// load the network file
	void loadnet(string sfile ="");

	// use LBFGS algorithm to train the model, 
	// Maybe some other SGD algorithm is better (faster)
	void _train_lbfgs();

	// The key attention phase, 
	// I will try my best to describe this part clearer!
	// The return value is our desirable SEMANTIC SCORE
	lbfgsfloatval_t bilattional_semantic(
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
		);
	// The back-propogation process of the `bilattional_semantic` procedure,
	// the data flows from the BiattSemValue into the words
	void bilattional_semantic_backprop(
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
		);

	// training or testing only one instance
	void train_a_instance(string instance, 
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
		lbfgsfloatval_t* grand);
	lbfgsfloatval_t test_a_instance(string src, string tgt, lbfgsfloatval_t* x);

	// tuning the model with respect the dev file
	lbfgsfloatval_t dev_tun(lbfgsfloatval_t* cX=NULL);

	// core evaluate
	lbfgsfloatval_t evaluate(
		const lbfgsfloatval_t *x,
	        lbfgsfloatval_t *g,
	        const int n,
	        const lbfgsfloatval_t step
	);
	static lbfgsfloatval_t _evaluate(
		void *instance,
	        const lbfgsfloatval_t *x,
	        lbfgsfloatval_t *g,
	        const int n,
	        const lbfgsfloatval_t step
	);
	int progress(
		const lbfgsfloatval_t *x,
	        const lbfgsfloatval_t *g,
	        const lbfgsfloatval_t fx,
	        const lbfgsfloatval_t xnorm,
	        const lbfgsfloatval_t gnorm,
	        const lbfgsfloatval_t step,
	        int n,
	        int k,
	        int ls
	);
	static int _progress(
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
	);

private:
	Parameter* para;					// parameters from config
	Vocabulary* vocab;					// vocabulary
	long file_lines;					// training instance number
	lbfgsfloatval_t* x;					// all model parameters

	int save_times;						// save numbers
	int best_model;						// the best model index
	lbfgsfloatval_t best_score;				// the best score
};

#endif
