#ifndef __VOCABULARY_H__
#define __VOCABULARY_H__

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <cstdlib>

#include "Parameter.h"
#include "Util.h"
#include "Constant.h"

using namespace std;

// vocabulary classes
class Vocabulary{
public:
	Vocabulary(Parameter& para);
	Vocabulary(){}

	// convert the string training files into id lists
	long convert_train_file(string train, vector<string>& procd);

	// save or load the vocabulary from files
	void save_vocab(string& save_vocab_file);
	void load_vocab(string load_vocab_file);

	// get the id of word
	long get_id(string& word, bool is_src);
	// get the word form this id
	string get_word(long& id);

	// get vocabulary size
	int get_source_size();
	int get_target_size();

private:
	void construct_vocab(string& file, string& oov_num);

	
public:
	map<string,long> word2id;		// word => id
	map<long,string> id2word;		// id => word 
private:
	int source_word_num;			// source words
	int target_word_num;			// target words
};

#endif
