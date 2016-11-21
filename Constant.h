#ifndef __CONSTANT_H__
#define __CONSTANT_H__

#include <iostream>
#include <lbfgs.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/* some constant definations */

// symbol defination
const string MOSES_SEP = " ||| ";			// Moses seperator
const string SPACE = " ";				// Space symbol
const string TAB = "\t";				// Tab symbol
const string OOV = "<<OOV>>";				// oov symbol

const string LFLAG    = "(";				// (
const string MFLAG    = ",";				// ,
const string RFLAG    = ")";				// )

const string SRC      = "1S_";				// source flag
const string TGT      = "2T_";				// target flag

// type defination
// define the vector and matrix for lbfgs
typedef Matrix<lbfgsfloatval_t,Eigen::Dynamic,Eigen::Dynamic> MatrixLBFGS;
typedef Matrix<lbfgsfloatval_t,Eigen::Dynamic,1> VectorLBFGS;

typedef pair<int, int> Span;				// alignment span

// utility marco
#define __FILE_MSG__(msg) 				\
{							\
	cerr << "mesg:\t" << msg      << endl;		\
	cerr << "file:\t" << __FILE__ << endl;		\
	cerr << "line:\t" << __LINE__ << endl;		\
	cerr << "func:\t" << __FUNCTION__ << endl;	\
}

#define __LBFGSALLOC__(name, size)			\
	name = lbfgs_malloc(size);	\
	if(name == NULL){				\
		__FILE_MSG__(				\
			"memory malloc failed!"		\
		);					\
		exit(1);				\
	}						\
	memset(name, 0, sizeof(lbfgsfloatval_t) * size);

#define __LBFGSFREE__(name)				\
	if(name != NULL) lbfgs_free(name);

#define __MEMALLOC__(type, name, size)			\
	type* name = 					\
		(type*)malloc(sizeof(type) * size);	\
	if(name == NULL){				\
		__FILE_MSG__(				\
			"memory malloc failed!" 	\
		);					\
		exit(1);				\
	}						\
	memset(name, 0, sizeof(type) * size);		
#define __MEMFREE__(name)				\
	if(name != NULL) free(name);

// for extracting some parameters
#define __DEFINE_WBS__(x, n)			\
	Map<MatrixLBFGS> m_W_1(x, n, 2 * n);	\
	Map<MatrixLBFGS> m_W_2(x + n * 2 * n, 2 * n, n);	\
	Map<VectorLBFGS> m_B_1(x + n * 2 * n + 2 * n * n, n); 	\
	Map<VectorLBFGS> m_B_2(x + n * 2 * n + 2 * n * n + n, 2 * n);

#define __DEFINE_DWBS__(x, n)			\
	Map<MatrixLBFGS> m_Dw_1(x, n, 2 * n);	\
	Map<MatrixLBFGS> m_Dw_2(x + n * 2 * n, 2 * n, n);	\
	Map<VectorLBFGS> m_Db_1(x + n * 2 * n + 2 * n * n, n); 	\
	Map<VectorLBFGS> m_Db_2(x + n * 2 * n + 2 * n * n + n, 2 * n);

#define __DEFINE_ATT_WBS__(x, n1, n2, n, s)	\
	Map<MatrixLBFGS> m_Aw_s(x, n, n1);	\
	Map<MatrixLBFGS> m_Aw_t(x + n * n1, n, n2);	\
	Map<VectorLBFGS> m_Aw_b(x + n * n1 + n * n2, n);	\
	Map<MatrixLBFGS> m_Sw_s(x + n * n1 + n * n2 + n, s, n1);	\
	Map<MatrixLBFGS> m_Sw_t(x + n * n1 + n * n2 + n + s * n1, s, n2);	\
	Map<VectorLBFGS> m_Sw_b(x + n * n1 + n * n2 + n + s * n1 + s * n2, s);	\
	Map<MatrixLBFGS> m_Sw(x + n * n1 + n * n2 + n + s * n1 + s * n2 + s, s, s); \
	Map<VectorLBFGS> m_Sb(x + n * n1 + n * n2 + n + s * n1 + s * n2 + s + s * s, 1);

#define __DEFINE_ATT_DWBS__(x, n1, n2, n, s)		\
	Map<MatrixLBFGS> m_DAw_s(x, n, n1);	\
	Map<MatrixLBFGS> m_DAw_t(x + n * n1, n, n2);	\
	Map<VectorLBFGS> m_DAw_b(x + n * n1 + n * n2, n);	\
	Map<MatrixLBFGS> m_DSw_s(x + n * n1 + n * n2 + n, s, n1);	\
	Map<MatrixLBFGS> m_DSw_t(x + n * n1 + n * n2 + n + s * n1, s, n2);	\
	Map<VectorLBFGS> m_DSw_b(x + n * n1 + n * n2 + n + s * n1 + s * n2, s);	\
	Map<MatrixLBFGS> m_DSw(x + n * n1 + n * n2 + n + s * n1 + s * n2 + s, s, s); \
	Map<VectorLBFGS> m_DSb(x + n * n1 + n * n2 + n + s * n1 + s * n2 + s + s * s, 1);

#endif
