/*
 * debug.h
 *
 *  Created on: May 13, 2009
 *      Author: Bhupendra Aole
 */

#ifndef DEBUG_H_
#define DEBUG_H_

#include <fstream>

using namespace std;

class cdebug {
private:
	ofstream myfile;
public:
	void open_debug_file();
	void debug (char *m);
	void debug (int n);
	void close_debug_file();
};

#endif /* DEBUG_H_ */
