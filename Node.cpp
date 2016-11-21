#include "Node.h"

Node::Node(lbfgsfloatval_t* vector, int dim, bool isLeaf){
	this -> isLeaf = isLeaf;
	this -> dim = dim;

	this -> id = OOV;
	this -> lChild = NULL;
	this -> rChild = NULL;

	this -> is_root = false;
	this -> span.first = 0, this -> span.second = 0; // initial [0,0]

	//{ allocate memory
	__LBFGSALLOC__(v_vector, dim);
	__LBFGSALLOC__(v_cerror, dim);
	__LBFGSALLOC__(v_deriva, dim * dim);
	__LBFGSALLOC__(v_oerror, 2 * dim);
	__LBFGSALLOC__(v_perror, 2 * dim);
	//}
	//{ deal with leaves
	if(isLeaf){//  for leaves, the gradient set to be I
		Map<VectorLBFGS>(v_vector, dim) = 
			Map<VectorLBFGS>(vector, dim);
		Map<MatrixLBFGS>(v_deriva, dim, dim).setIdentity();
	}else{
		Map<VectorLBFGS>(v_vector, dim) 
			= Map<VectorLBFGS>(vector, dim);
		Map<MatrixLBFGS>(v_deriva, dim, dim).setZero();
	}
	//}
}

Node::~Node(){
	// recusive deleting
	if(this -> lChild != NULL) delete this -> lChild;
	if(this -> rChild != NULL) delete this -> rChild;

	//{ free memory
	__LBFGSFREE__(v_vector);
	__LBFGSFREE__(v_cerror);
	__LBFGSFREE__(v_deriva);
	__LBFGSFREE__(v_oerror);
	__LBFGSFREE__(v_perror);
	//}
}
