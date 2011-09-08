/*
 * board.h
 *
 *  Created on: Sep 4, 2011
 *      Author: Bhupendra Aole
 */

#ifndef BOARD_H_
#define BOARD_H_

#include "constants.h"
#include "bitmove.h"

class board {
public:
	static const int MAX_MOVES = 300;

	int isready;

	side moveof;
	side movenotof;

	bitboard all[2]; 		// [side]
	bitboard pieces[2][6]; 	// [side][piece]

	int currentmovenumber;
	int capturedpiece[MAX_MOVES]; // [currentmovenumber]
	bitboard history[MAX_MOVES]; 	// [currentmovenumber]
	int kingmovedat[2]; 	// [side]
	int rookmovedat[2][2]; 	// [side][0:a file, 1:h file]
	int isdoublemove[MAX_MOVES];
	bitboard enpasquare[MAX_MOVES];

public:
	board();
	void newgame();
	void domove(bitmove m);
	void undmove(bitmove m);
	void undolastmove();
};

#endif /* BOARD_H_ */
