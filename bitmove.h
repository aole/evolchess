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
public:
	bitmove(bitboard m) { move = m; };
};

#endif /* BITMOVE_H_ */
