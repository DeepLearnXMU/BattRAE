#ifndef __NODE_H__
#define __NODE_H__

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>

#include "Constant.h"
#include "Util.h"

using namespace std;
using namespace Eigen;

// The node on a tree
// This class acts like a struct object, 
// this is because the tree access this class very frequently,
// making all members public is very convenient
// maybe this class should have more function ...
class Node{
public:
	Node(lbfgsfloatval_t* vector, int dim, bool isLeaf);
	~Node();

	bool isLeaf;				// is Leaf Node?
	bool is_root;				// is Root Node?
	Span span;				// Span In Sentence

	string   id;				// Sentences ID Presentation
	int 	dim;				// Word Dimension

	Node* lChild;				// Children Node, left
	Node* rChild;				// Children Node, Right

	lbfgsfloatval_t* v_vector;		// Current Representation at this Node
	lbfgsfloatval_t* v_cerror;		// Current Errors at this Node
	lbfgsfloatval_t* v_deriva;		// Current Derivation at this Node
	lbfgsfloatval_t* v_oerror;		// Error at Re-construction
	lbfgsfloatval_t* v_perror;		// Errors From Parent Nodes
};

#endif
