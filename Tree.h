#ifndef __TREE_H__
#define __TREE_H__

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
#include "Node.h"

using namespace std;
using namespace Eigen;

// This class contains the forprop and backprop of tree structures
// the key part for recursive autoencoder
class Tree{
public:
	Tree(string phrase
		, int dim
		, int base
		, bool istree
		, lbfgsfloatval_t alpha
		, lbfgsfloatval_t* words
		, lbfgsfloatval_t* theta
		, lbfgsfloatval_t* gWord
		, lbfgsfloatval_t* gTheta);
	~Tree();

	// build tree structure from the given structure
	void build_from_tree();
private:
	Node* _build_from_tree(vector<string>& v_str);

public:
	// key procedure for tree construction
	// build tree structure from the given plain string
	void build_to_tree();

	// get the string description of this tree
	string print_tree();
private:
	string _print(Node* root);
	
public:
	// backpropagation
	void backprop();
private:
	void _back_prop(Node* root, lbfgsfloatval_t* theta,
			lbfgsfloatval_t* gWord, lbfgsfloatval_t* gTheta);

public:
	Node* root;						// The Root Node in this Tree
	vector<Node*> nodes;					// Nodes Set<Vectors>, Important for Classification
	lbfgsfloatval_t rfValue;				// Reconstruction Errors
private:
	int dim;						// Word Dimension
	int size;						// tree size
	lbfgsfloatval_t alpha;					// Balance Factor, for Reconstruction

	string phrase;						// Phrase Strings
	int base;						// The base index relative to the source phrase

	lbfgsfloatval_t* words;					// Words Parameters
	lbfgsfloatval_t* theta;					// Theta Parameters
	lbfgsfloatval_t* gWord;					// Words Gradients
	lbfgsfloatval_t* gTheta;				// Theta Gradients
};
//}

#endif
