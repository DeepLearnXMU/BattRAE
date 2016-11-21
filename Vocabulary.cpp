#include "Vocabulary.h"

Vocabulary::Vocabulary(Parameter& para){
	string save_vocab_file = para.get_para("[save_vocab]");
	string corps_file = para.get_para("[train_file]");
	string oov_num = para.get_para("[oov_number]");

	cout << "#Construct Vocabulary" << endl;
	construct_vocab(corps_file, oov_num);
	cout << "#Save Vocabulary" << endl;
	save_vocab(save_vocab_file);
}

// maybe this design is not so well ...:(
long Vocabulary::convert_train_file(string train, vector<string>& procd){
	ifstream is(train.c_str());
	if(!is){
		__FILE_MSG__(
			"failed to read:\t" << "\"" << train << "\""
		);
		return 0;
	}

	string line = "", conv = "";
	vector<string> v_str, v_phr;
	long count = 0;
	while(getline(is, line)){
		if((line = strip_str(line)) == "") continue;

		++ count;
		v_phr = split_str(line, MOSES_SEP), conv = "";
		if(v_phr.size() != 4){
			__FILE_MSG__(
				"Bad File Format, this should be: 'src ||| tgt ||| neg src ||| neg tgt'" << endl
				<< "But Now:" << "\"" << line << "\""
			);
			exit(1);
		}

		// src
		v_str = split_str(v_phr[0], SPACE);
		for(size_t j = 0; j < v_str.size(); ++ j){
			conv += num2str(get_id(v_str[j], true)) + SPACE;
		}
		conv = strip_str(conv) + MOSES_SEP;
		// tgt
		v_str = split_str(v_phr[1], SPACE);
		for(size_t j = 0; j < v_str.size(); ++ j){
			conv += num2str(get_id(v_str[j], false)) + SPACE;
		}
		conv = strip_str(conv) + MOSES_SEP;
		// neg src
		v_str = split_str(v_phr[2], SPACE);
		for(size_t j = 0; j < v_str.size(); ++ j){
			conv += num2str(get_id(v_str[j], true)) + SPACE;
		}
		conv = strip_str(conv) + MOSES_SEP;
		// neg tgt
		v_str = split_str(v_phr[3], SPACE);
		for(size_t j = 0; j < v_str.size(); ++ j){
			conv += num2str(get_id(v_str[j], false)) + SPACE;
		}
		procd.push_back(strip_str(conv));

		if(count % 10000 == 0){
			cout << "Converting the training file:\t" << count << endl;
		}
	}
	is.close();

	return count;
}

void Vocabulary::construct_vocab(string& file, string& oov_num){
	ifstream is(file.c_str());
	if(!is){
		__FILE_MSG__(
			"failed to read:\t" << "\"" << file << "\""
		);
		exit(1);
	}

	//{
	string line = "";
	vector<string> v_str, v_sub_str;
	map<string, int> wordmap;		
	long count = 0;
	while(getline(is, line)){
		if((line = strip_str(line)) == "") continue;

		v_str = split_str(line, MOSES_SEP);
		if(v_str.size() != 4){
			__FILE_MSG__(
				"Bad File Format, this should be: 'src ||| tgt ||| neg src ||| neg tgt'" << endl
				<< "But Now:" << "\"" << line << "\""
			);
			exit(1);
		}

		// Correct source phrase
		v_sub_str = split_str(v_str[0], SPACE);
		for(size_t j = 0; j < v_sub_str.size(); ++ j){
			wordmap[SRC + v_sub_str[j]] ++;
		}
		// Correct target phrase
		v_sub_str = split_str(v_str[1], SPACE);
		for(size_t j = 0; j < v_sub_str.size(); ++ j){
			wordmap[TGT + v_sub_str[j]] ++;
		}

		++ count;
		if(count % 10000 == 0){
			cout << "Constructing the vocabulary:\t" << count << endl;
		}
	}
	is.close();
	//}

	//{
	long id = 0;
	int oov = atoi(oov_num.c_str());
	source_word_num = 0;
	target_word_num = 0;
	wordmap[SRC + OOV] = 1000000;
	wordmap[TGT + OOV] = 1000000;
	for(map<string, int>::iterator m_it = wordmap.begin(); m_it != wordmap.end(); ++ m_it){
		// if the word count less equal the OOV setting, disable this word
		if(m_it -> second <= oov) continue;	// OOV
		if(word2id.find(m_it -> first) != word2id.end()) continue;

		word2id[m_it -> first] = id;
		id2word[id] = m_it -> first;
		++ id;

		if(m_it->first.find(SRC) == 0){
			++ source_word_num;
		}else{
			++ target_word_num;
		}
	}
	//}
}

void Vocabulary::save_vocab(string& save_vocab_file){
	ofstream os(save_vocab_file.c_str());

	// save into `word\tid`
	map<string, long>::iterator m_it;
	for(m_it = word2id.begin(); m_it != word2id.end(); ++ m_it){
		os << m_it -> first << TAB << m_it -> second << endl;
	}
	os.close();
}
void Vocabulary::load_vocab(string load_vocab_file){
	word2id.clear(), id2word.clear();
	
	ifstream is(load_vocab_file.c_str());
	if(!is){
		__FILE_MSG__(
			"failed to read:\t" << "\"" << load_vocab_file << "\""
		);
		exit(1);
	}
	
	// load from `word\tid`
	string line = ""; vector<string> v_str;
	source_word_num = 0; target_word_num = 0;
	while(getline(is, line)){
		if((line = strip_str(line)) == "") continue;

		v_str = split_str(line, TAB);
		if(v_str.size() != 2){
			__FILE_MSG__(
				"bad file format:\t" << "\"" << line << "\""
			);
			exit(1);
		}
		
		string word = v_str[0];
		long id = atoi(v_str[1].c_str());
		word2id[word] = id;
		id2word[id] = word;

		if(word.find(SRC) == 0){
			++ source_word_num;
		}else{
			++ target_word_num;
		}
	}
	is.close();
}

// get the id of word
long Vocabulary::get_id(string& word, bool is_src){
	string prefix = SRC;
	if(!is_src) prefix = TGT;

	map<string, long>::iterator m_it = word2id.find(prefix + word);
	if(m_it == word2id.end()) return word2id[prefix + OOV];
	return m_it -> second;
}

// get the word form this id
string Vocabulary::get_word(long& id){
	map<long, string>::iterator m_it = id2word.find(id);
	if(m_it == id2word.end()) return OOV;
	return m_it -> second.substr(3);
}

// get vocabulary size
int Vocabulary::get_source_size(){
	return source_word_num;
}
int Vocabulary::get_target_size(){
	return target_word_num;
}
