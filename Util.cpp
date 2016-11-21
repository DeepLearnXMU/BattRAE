#include "Util.h"

string strip_str(const string& s, const string& chs)
{
	if (s.empty()) {
		return s;
	}

	int i = 0;
	while (i < (int)s.size() && chs.find(s[i]) != string::npos) {
		++i;
	}

	int j = (int)s.size() - 1;
	while (j >= 0 && chs.find(s[j]) != string::npos) {
		--j;
	}

	++j;

	string ret =  "";
        if (i < j) ret = s.substr(i, j - i);
	return ret;
};

vector<string> split_str(const string& i_str, const string& input_delimit)
{
	vector<string> v_str;
	string input_str = strip_str(i_str);
	int len_sep=(int)input_delimit.length();

	size_t pre_pos=0;
	size_t pos=input_str.find(input_delimit, pre_pos);

	while(pos!=string::npos)
	{		
		int temp_len=pos-pre_pos;
		if(temp_len!=0)
		{
			v_str.push_back(input_str.substr(pre_pos, temp_len));
		}else{
			v_str.push_back("");
		}

		pre_pos=pos+len_sep;
		pos=input_str.find(input_delimit, pre_pos);
	}

	if((int)input_str.length()>(int)pre_pos)
	{
		v_str.push_back(input_str.substr(pre_pos,(int)input_str.length()));
	}

	return v_str;
}

string join_str(const vector<string>& input_v_word, const string& input_sep)
{
	string str;
	for (size_t i = 0; i < input_v_word.size(); i++) {
		str += input_v_word[i];
		if (i != input_v_word.size() - 1) { 
			str += input_sep; 
		}
	}

	return str;
};

string dou2str(const double& d){
	stringstream ss;
	ss<<d;
	return ss.str();
};

string num2str(const long& i){
	stringstream ss;
	ss<<i;
	return ss.str();
}

string remove_sides(const string& str)
{
	if (str.size()<=2) {
		cout << "Remove_side error, fail to remove sides, the str is: \"" << str << "\"" << endl;
		exit(1);
	}

	return str.substr(1, (int)str.size()-2);
};

string replace_word(const string& tree, const string& ids){
	vector<string> v_trees = split_str(tree, " ");
	vector<string> v_ids = split_str(ids, " ");

	string rp_str = "";
	size_t j = 0;
	for(size_t i = 0; i < v_trees.size(); ++ i){
		if(v_trees[i] == "(" || v_trees[i] == ")" || v_trees[i] == ","){
			rp_str += v_trees[i] + " ";
		}else{
			vector<string> v_tmp = split_str(v_trees[i], ":");
			rp_str += v_ids[j] + ":" + v_tmp[1] + " ";
			++ j;
		}
	}
	return strip_str(rp_str);
}
