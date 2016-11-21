#ifndef __UTIL_H__
#define __UTIL_H__

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <stdlib.h>

using namespace std;

/* Some utility functions  */

// remove the space symbol on both sides
string strip_str(const string& s, const string& chs="\t ");

// split string into a vector using the delimit
vector<string> split_str(const string& i_str, const string& input_delimit);

// combine strings within a vector into a single string
string join_str(const vector<string>& input_v_word, const string& input_sep = " ");

// convert double to string
string dou2str(const double& d);

// convert long to string
string num2str(const long& i);

// remove some un-needed symbols, mainly used for Config.ini
string remove_sides(const string& str);

// replace the words in the tree string with the given new ids
string replace_word(const string& tree, const string& ids);

#endif
