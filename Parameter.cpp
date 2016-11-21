#include "Parameter.h"


Parameter::Parameter(const string& config_file) {
	string line, title_str,value_str;

	ifstream is(config_file.c_str());
	if(!is) { 
		cerr << "Parameter Error, fail to read config file from \"" << config_file << "\"." << endl;
		exit(1);
	}

	while (getline(is, line)) {
		// if empty line or the line starts with ###, skip
		// ### is the default comment line
		if (line == "" || line.find("###")!= string::npos) { 
			continue; 
		}

		istringstream iss(line.c_str());
		title_str = ""; value_str = "";
		iss >> title_str >> value_str;
		m_para[title_str] = value_str;
	}

	is.close();
}

// extract the corresponding value with the give key
string Parameter::get_para(const string& title) {
	map<string, string>::iterator m_it = m_para.find(title);
	if (m_it == m_para.end()) {
		cerr << "GetPara Error, fail to read the value of item \"" << title << "\"." << endl;
		cerr << "The title_str of item is: " << title << endl;
		exit(1);
	}
	
	return m_it->second;
}


// re-set some keys with new values if necessary
void Parameter::set_para( const string& input_item, const string& input_value_str ) {
	m_para[input_item] = input_value_str;
}
