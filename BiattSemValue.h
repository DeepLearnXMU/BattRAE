#ifndef __BIATTSEMVALUE_H__
#define __BIATTSEMVALUE_H__

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>

#include <lbfgs.h>
#include <Eigen/Dense>

#include "Constant.h"
#include "Util.h"
#include "Tree.h"

using namespace std;
using namespace Eigen;

// like the node class, this is a struct for computation
class BiattSemValue{
public:
	Tree* src_tree;
	Tree* tgt_tree;
	bool is_tree;

	lbfgsfloatval_t semScore;

	lbfgsfloatval_t* theta_src_att_mat;	// S * n
	lbfgsfloatval_t* theta_tgt_att_mat;	// T * n
	lbfgsfloatval_t* grand_src_att_mat;	// S * n
	lbfgsfloatval_t* grand_tgt_att_mat;	// T * n

	lbfgsfloatval_t* grand_src_att_score;	// S * S
	lbfgsfloatval_t* grand_tgt_att_score; 	// T * T
	lbfgsfloatval_t* theta_src_att_score;	// S * 1
	lbfgsfloatval_t* theta_tgt_att_score;	// T * 1

	lbfgsfloatval_t* theta_src_rae_mat;	// S * n1
	lbfgsfloatval_t* theta_tgt_rae_mat;	// T * n2

	lbfgsfloatval_t* grand_src_sem_rep;	// ns * 1
	lbfgsfloatval_t* grand_tgt_sem_rep;	// nt * 1
	lbfgsfloatval_t* theta_src_sem_rep;	// ns * 1
	lbfgsfloatval_t* theta_tgt_sem_rep;	// nt * 1

	lbfgsfloatval_t* theta_src_inst_representation; 	// n1 * 1
	lbfgsfloatval_t* theta_tgt_inst_representation;		// n2 * 1

	lbfgsfloatval_t* biattention_matrix;			// ns * nt
	lbfgsfloatval_t* grand_biattention_matrix;		// ns * nt

	lbfgsfloatval_t* grand_v_score;		// 1 * 1

	BiattSemValue(bool is_tree);
	~BiattSemValue();
};

#endif
