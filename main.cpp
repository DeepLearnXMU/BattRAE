#include <iostream>

#include "Parameter.h"
#include "Vocabulary.h"
#include "BattRAE.h"
#include "Constant.h"

int main(int argc, char** argv){
	if(argc != 3){
		__FILE_MSG__(
			"Usage: ./" << argv[0] << " config_file -train|-test"
		);
		exit(1);
	}

	srand(time(NULL));
	Eigen::initParallel();

	string cfg = argv[1];
	Parameter para(cfg);
	Vocabulary* vocab = NULL;

	// this may speed up the matrix operation
	Eigen::setNbThreads(atoi(para.get_para("[thread_num]").c_str()));
	string mode = argv[2];

	if(mode != "-train"){
		vocab = new Vocabulary();
		vocab -> load_vocab(para.get_para("[save_vocab]"));
	}else{
		vocab = new Vocabulary(para);
	}

	BattRAE barae(&para, vocab);

	if(mode == "-train"){
		barae.train();
	}else{
		barae.test();
	}

	delete vocab;
	return 0;
}
