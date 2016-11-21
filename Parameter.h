#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <cstdlib>

using namespace std;

/*
 Read and record the config information from `Config.ini`
 */
class Parameter
{
public:

	// read key-values from the config_file
	Parameter(const string& config_file);

	// get parameters using keys
	string get_para(const string& title);

	// reset corresponding key-values
	void set_para( const string& input_item, const string& input_value_str);

public:
	map<string, string> m_para;
};

#endif
