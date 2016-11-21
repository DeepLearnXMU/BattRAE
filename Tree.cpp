#include "Tree.h"

Tree::Tree(string phrase
	, int dim
	, int base
	, bool istree
	, lbfgsfloatval_t alpha
	, lbfgsfloatval_t* words
	, lbfgsfloatval_t* theta
	, lbfgsfloatval_t* gWord
	, lbfgsfloatval_t* gTheta){
	this -> phrase = phrase;
	this -> words = words;
	this -> theta = theta;
	this -> dim = dim;
	this -> alpha = alpha;
	this -> base = base;

	this -> rfValue = 0.0;

	this -> gWord = gWord;
	this -> gTheta = gTheta;

	vector<string> v_str = split_str(phrase, SPACE);

	if(!istree){
		build_to_tree();
	}else{
		// adjust the alpha according to the tree size
		// Or simply set it to 1?
		this -> size = (int) ((v_str.size() + 3) / 4.0);
		if(this -> size <= 0) this -> size = 1;
		this -> alpha /= size;
		build_from_tree();
	}
}
Tree::~Tree(){
	if(root != NULL) delete root;
	this->nodes.clear();
}

void Tree::build_from_tree(){
	vector<string> v_str = split_str(phrase, SPACE);
	root = _build_from_tree(v_str);
	root -> is_root = true;
}

Node* Tree::_build_from_tree(vector<string>& v_str){
	// create the non-leaf node
	if(v_str[0] == "("){
	        v_str.erase(v_str.begin());                     // (
		Node* lChild = _build_from_tree(v_str);
	        v_str.erase(v_str.begin());                     // ,
		Node* rChild = _build_from_tree(v_str);
	        v_str.erase(v_str.begin());                     // )

		//{ extract the parameter w&b structure
		__DEFINE_WBS__(theta, dim);
		//}
				
		//{ build up parameters
		VectorLBFGS m_pbuild(2 * dim);					// childs raw input
		VectorLBFGS m_rbuild(2 * dim);					// childs re-builds
		VectorLBFGS m_parent(dim);					// parent node represent

		VectorLBFGS m_pt_sech(dim);					// parent sech
		VectorLBFGS m_pt_tanh(dim);					// parent tanh
		VectorLBFGS m_rd_sech(2 * dim);					// rebuild sech
		VectorLBFGS m_rd_tanh(2 * dim);					// rebuild tanh

		VectorLBFGS m_error(2 * dim);					// current error
		MatrixLBFGS m_deriva(dim, dim);					// current derivation
		//}

		m_pbuild.segment(0,dim) = Map<VectorLBFGS>(lChild -> v_vector, dim);
		m_pbuild.segment(dim,dim) = Map<VectorLBFGS>(rChild -> v_vector, dim);

		//{ <x1, x2> => p
		// z1 = W1x + b1
		m_parent = m_W_1 * m_pbuild + m_B_1;
		// sech & tanh
		m_pt_sech = 2.0 / (m_parent.array().exp() + (-1 * m_parent).array().exp());
		m_pt_tanh = (m_parent.array().exp() - (-1 * m_parent).array().exp()).array() 
			/ (m_parent.array().exp() + (-1 * m_parent).array().exp()).array();
		// a1 = f(z1), where f = tanh
		m_parent = (m_parent.array().exp() - (m_parent * -1).array().exp()).array()
			/ (m_parent.array().exp() + (m_parent * -1).array().exp()).array();
		// normalize
		m_parent /= m_parent.norm();
		//}

		//{ p => <x1',x2'>
		// z2 = W2p + b2
		m_rbuild = m_W_2 * m_parent + m_B_2;
		// a2 = f(z2), where f = tanh
		m_rbuild = (m_rbuild.array().exp() - (m_rbuild * -1).array().exp()).array()
			/ (m_rbuild.array().exp() + (m_rbuild * -1).array().exp()).array();
		//}
			
		//{ objective
		lbfgsfloatval_t fx = (m_pbuild - m_rbuild).array().square().sum() * 1/2 * alpha;
		//}

		//{ build up the parent node
		Node* parent = new Node(m_parent.data(),dim, false);
		parent -> id = lChild -> id + " " + rChild -> id;
		parent -> span.first = lChild -> span.first;
		parent -> span.second = rChild -> span.second;
		//}

		//{ tanh f' = 1 - f^2
		// compute the derivations
		m_deriva  = 
			m_pt_sech.array().square().matrix().asDiagonal()
			* ((1 / m_pt_tanh.norm()) * MatrixLBFGS::Identity(dim, dim).array() - (m_pt_tanh * m_pt_tanh.transpose() / pow(m_pt_tanh.norm(), 3.0)).array()).matrix()  			// tanh^2
			;
		Map<MatrixLBFGS>(parent -> v_deriva, dim, dim) = m_deriva;
		//}

		//{ e2 = (a2 - y)*f'(z2)
		m_error = alpha * (m_rbuild - m_pbuild).array(); // a2 - y 
		
		VectorLBFGS m_D_error = 1 - m_rbuild.array().square();
		m_error = (m_error.array() * m_D_error.array()).matrix();
		Map<VectorLBFGS>(parent -> v_perror, 2 * dim) = m_error;
		VectorLBFGS o_err = (m_pbuild - m_rbuild);
		Map<VectorLBFGS>(parent -> v_oerror, 2 * dim) = o_err;
		//}

		//{ collect these errors
		rfValue += fx;
		//}

		//{ relate the children
		parent -> lChild = lChild;
		parent -> rChild = rChild;
		//}
				   
		//{ save the internal non-leaf nodes for attention
		nodes.push_back(parent);	
		//}
	        return parent;
	}
	// illegal cases
	else if(v_str[0] == "," || v_str[0] == ")"){
		__FILE_MSG__(
			"bad tree tructure!" << endl <<
			"tree:\t" << "\"" << v_str[0] << "\""
		);
	        exit(1);
	}

	// create a leaf node
	size_t pos = v_str[0].find(':');
	if(pos == v_str[0].npos){
		__FILE_MSG__(
			"bad tree format:\t" << "\"" << v_str[0] << "\"" << endl
			<< "there should be a ':' to split the word_id and the word_pos"
		);
		exit(1);
	}
	string id = v_str[0].substr(0, pos);
	string locat = v_str[0].substr(pos + 1);
  	Node* child = new Node(
			words + dim * (atoi(id.c_str()) - base), 
			dim, 
			true); 
	child -> id = id;
	child -> span.first = atoi(locat.c_str());
	child -> span.second = atoi(locat.c_str());

	//{ save the leaf nodes for attention
	nodes.push_back(child);
	//}

        v_str.erase(v_str.begin());
        return child;
}

void Tree::build_to_tree(){
	vector<string> v_str = split_str(phrase, SPACE);

	//{ parameters
	VectorLBFGS v_pbuild(2 * dim);
	VectorLBFGS v_parent(dim);
	VectorLBFGS v_rbuild(2 * dim);
	//}
	//{ extract the W&Bs
	__DEFINE_WBS__(theta, dim);
	//}

	vector<Node*> v_nodes;
	for(size_t i = 0; i < v_str.size(); ++ i){
		Node* leaf = new Node(words + dim * (atoi(v_str[i].c_str()) - base), dim, true); 
		leaf -> id = v_str[i];
		leaf -> span.first = (int)i, leaf -> span.second = (int)i;
		v_nodes.push_back(leaf);
		nodes.push_back(leaf);
	}
	// nessary internal parameters
	Node* parent = NULL;			// parent nodes
	VectorLBFGS s_parent(dim);		// save the parent representation
	size_t l = 0, r = 1;			// left&right parameters
	lbfgsfloatval_t minValue = 1E8;		// the minimize values for tree construction

	// bottom up, greedily
	for(size_t i = 1; i < v_str.size(); ++ i){ 
		// find the minimum neighboring nodes
		minValue = 1E8;
		parent = NULL;
		l = 0, r = 1;
		for(size_t j = 0; j < v_nodes.size() - 1; ++ j){
			v_pbuild.segment(0,dim) = Map<VectorLBFGS>(v_nodes[j] -> v_vector, dim);
			v_pbuild.segment(dim,dim) = Map<VectorLBFGS>(v_nodes[j + 1] -> v_vector, dim);

			//{ <x1, x2> => p
			// z1 = W1x + b1
			v_parent = m_W_1 * v_pbuild + m_B_1;

			// a1 = f(z1), where f = tanh
			v_parent = (v_parent.array().exp() - (v_parent * -1).array().exp()).array()
				/ (v_parent.array().exp() + (v_parent * -1).array().exp()).array();
			// normalize
			v_parent /= v_parent.norm();
			//}

			//{ p => <x1',x2'>
			// z2 = W2p + b2
			v_rbuild = m_W_2 * v_parent + m_B_2;
			// a2 = f(z2), where f = tanh
			v_rbuild = (v_rbuild.array().exp() - (v_rbuild * -1).array().exp()).array()
				/ (v_rbuild.array().exp() + (v_rbuild * -1).array().exp()).array();
			//}
			
			// objective
			lbfgsfloatval_t object = (v_pbuild - v_rbuild).array().square().sum();
			// compose the node pair whose reconstruction error is the smallest
			if(object < minValue){
				minValue = object;
				s_parent = v_parent;
				l = j;
				r = j + 1;
			}
		}

		// construct the parent nodes
		parent = new Node(s_parent.data(),dim, false);
		// relate the childrens
		parent -> lChild = v_nodes[l];
		parent -> rChild = v_nodes[r];
		parent -> span.first = v_nodes[l] -> span.first;
		parent -> span.second = v_nodes[r] -> span.second;
		nodes.push_back(parent);

		// continue
		v_nodes.erase(v_nodes.begin() + l, v_nodes.begin() + r + 1);
		v_nodes.insert(v_nodes.begin() + l, parent);
	}

	// checking
	if(v_nodes.size() != 1){
		__FILE_MSG__(
			"recursive tree build up error!" << endl << 
			"the size of v_nodes should be 1, but here \"" << v_nodes.size() << "\""
		);
		exit(1);
	}
	root = v_nodes[0];
	root -> is_root = true;
}

string Tree::print_tree(){
	string tree = _print(root);
	return tree;
}
string Tree::_print(Node* root){
	if(root == NULL) return "";
	if(root -> isLeaf){
		if(root -> span.first != root -> span.second){
			__FILE_MSG__(
				"bad not format:\t" << "\"" << root -> span.first << " != " << root -> span.second
				<< endl << "Note the leaf node must satisfy span.first = span.second"
			);
			exit(1);
		}
		return root -> id + ":" + num2str(root -> span.first);
	}

	string str = "";
	str += "( ";
	str += _print(root -> lChild);
	str += " , ";
	str += _print(root -> rChild);
	str += " )";

	return str;
}
	
void Tree::backprop(){
	_back_prop(this -> root, theta, gWord, gTheta);
}
void Tree::_back_prop(Node* root, lbfgsfloatval_t* theta,
		lbfgsfloatval_t* gWord, lbfgsfloatval_t* gTheta){
	//{ define the parameters
	__DEFINE_WBS__(theta, dim);
	__DEFINE_DWBS__(gTheta, dim);
	//}
	if(root -> isLeaf){	
		Map<VectorLBFGS>(gWord + dim * (atoi(root -> id.c_str()) - base), dim)
			+= Map<VectorLBFGS>(root -> v_cerror, dim);
	       	return ;
	}

	Map<VectorLBFGS> m_E_2(root -> v_perror, 2 * dim);				// E2 => 
	Map<VectorLBFGS> m_O_2(root -> v_oerror, 2 * dim);				// Eo => 
	Map<VectorLBFGS> m_E_1(root -> v_cerror, dim);					// E1 => 
	Map<VectorLBFGS> m_P_1(root -> v_vector,dim);					// P1 => 
	Map<MatrixLBFGS> m_D_f(root -> v_deriva, dim, dim);				// Df =>
	VectorLBFGS 	 m_P_0(2 * dim);						// P0 => 
	VectorLBFGS 	 m_E_0(2 * dim);						// E0 => 
	VectorLBFGS 	 back_E_0(2 * dim);						// E0 
	
	//{ bakprop E_2 => E_1, update Dw_2, Db_2
	m_E_1 += m_D_f * (m_W_2.transpose() * m_E_2);
	m_Dw_2 += m_E_2 * m_P_1.transpose();
	m_Db_2 += m_E_2;
	//}
	//{ backprop E_1 => E_0, update Dw_1, Db_1
	Map<MatrixLBFGS> m_D_0_l(root -> lChild -> v_deriva, dim, dim);
	Map<MatrixLBFGS> m_D_0_r(root -> rChild -> v_deriva, dim, dim);
	m_P_0.segment(0,dim) = Map<VectorLBFGS>(root -> lChild -> v_vector, dim);
	m_P_0.segment(dim,dim) = Map<VectorLBFGS>(root -> rChild -> v_vector, dim);
	back_E_0 = m_W_1.transpose() * m_E_1;
	back_E_0 += m_O_2 * alpha;
	m_E_0.segment(0, dim) = m_D_0_l * back_E_0.segment(0, dim);
	m_E_0.segment(dim, dim) = m_D_0_r * back_E_0.segment(dim, dim);
	m_Dw_1 += m_E_1 * m_P_0.transpose();
	m_Db_1 += m_E_1;
	//}

	//{ backprop to child
	if(root -> lChild == NULL || root -> rChild == NULL){
		__FILE_MSG__(
			"oh! my god, what's wrong with you, my phrase!!!!" << endl << 
			"error in TreeNode, error with NULL (lChild or rChild) !!!"
		);
		exit(1);
	}
	Map<VectorLBFGS>(root -> lChild -> v_cerror, dim) += m_E_0.segment(0, dim);
	Map<VectorLBFGS>(root -> rChild -> v_cerror, dim) += m_E_0.segment(dim, dim);
	_back_prop(root -> lChild, theta, gWord, gTheta);
	_back_prop(root -> rChild, theta, gWord, gTheta);
	//}
}
