/*
 * bitmove.h
 *
 *  Created on: Sep 4, 2011
 *      Author: baole
 */

#ifndef BITMOVE_H_
#define BITMOVE_H_

#include "constants.h"

#include<iostream>
#include <cstring>

using namespace std;

class bitmove {
public:
	bitboard from, to;
	piece promto;

public:
	bitmove() {
		from = to = 0;
		promto = none;
	}
	bitmove(bitboard f, bitboard t, piece pt = none) {
		from = f;
		to = t;
		promto = pt;
	}
	void copy(const bitmove &m) {
		from = m.from;
		to = m.to;
		promto = m.promto;
	}
	int set(const char *m);

	friend ostream &operator<<(ostream &s, bitmove m);
};


#endif /* BITMOVE_H_ */
