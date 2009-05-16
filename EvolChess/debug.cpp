/*
 * debug.cpp
 *
 *  Created on: May 13, 2009
 *      Author: baole
 */

#include <iostream>
#include "debug.h"

void cdebug::open_debug_file() {
  myfile.open ("D:/test.txt");
}

void cdebug::debug (char *m) {
  //myfile << m;
  cout << m;
}

void cdebug::debug (int n) {
	//myfile << n;
	cout << n;
}
void cdebug::close_debug_file() {
  myfile.close();
}
