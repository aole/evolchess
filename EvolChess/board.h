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
	 // TODO: change to dynamic array/stack
	static const int MAX_MOVES = 300;

	int isready;

	// side to move
	side moveof;
	side movenotof;

	// position
	bitboard pall[2];
	bitboard ppieces[2][6];

	/* castling rights */
	// king moved at which move
	int kingmovedat[2];
	// rooks moved at which move
	// 0: rook on a file; 1: rook on h file
	int rookmovedat[2][2];

	// en passant square
	// will be set to square just before to what a pawn moved to
	// only set in case of double pawn move else set to 0
	// array is useful when undoing moves
	bitboard enpasqr[MAX_MOVES];
	bitboard enpapwn[MAX_MOVES];

	// moves history
	// 0 when game starts
	int movenumber;
	bitmove history[MAX_MOVES];
	// which piece got captured at that move
	piece captured[MAX_MOVES];

public:
	board();
	~board() {};

	void newgame();
	void domove(const bitmove &m);
	void undolastmove();

	friend ostream &operator<<(ostream &s, board b);
};

#endif /* BOARD_H_ */
