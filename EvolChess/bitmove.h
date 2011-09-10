/*
 * bitmove.h
 *
 *  Created on: Sep 4, 2011
 *      Author: baole
 */

#ifndef BITMOVE_H_
#define BITMOVE_H_

#include "constants.h"

class bitmove {
public:
	bitboard move;
	piece promto;

public:
	bitmove() {
		move = 0;
		promto = none;
	}
	bitmove(bitboard m, piece pt = none) {
		move = m;
		promto = pt;
	}
	void copy(const bitmove &m) {
		move = m.move;
		promto = m.promto;
	}
};

#endif /* BITMOVE_H_ */
