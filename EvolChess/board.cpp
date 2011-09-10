/*
 * board.cpp
 *
 *  Created on: Sep 4, 2011
 *      Author: baole
 */

#include "board.h"

board::board() {
#ifdef DEBUG
	isready = 0;
#endif
}

void board::newgame() {
	// initialize to a standard new game
	movenumber = 0;
	moveof = white;
	movenotof = black;

	kingmovedat = {0,0};
	rookmovedat = { {0,0}, {0,0}};

#ifdef DEBUG
	isready = 1;
#endif
}

void board::domove(const bitmove &m) {
	bitboard from = m.move & pall[moveof];
	bitboard to = m.move ^ from;

	// adjust all pieces bitboard
	pall[moveof] ^= m.move;

	// get the pieces involved
	piece p = none, c = none;
	if (ppieces[moveof][king] & from)
		p = king;
	else if (ppieces[moveof][queen] & from)
		p = queen;
	else if (ppieces[moveof][rook] & from)
		p = rook;
	else if (ppieces[moveof][bishop] & from)
		p = bishop;
	else if (ppieces[moveof][knight] & from)
		p = knight;
	else if (ppieces[moveof][pawn] & from)
		p = pawn;

	if (ppieces[movenotof][queen] & to)
		c = queen;
	else if (ppieces[movenotof][rook] & to)
		c = rook;
	else if (ppieces[movenotof][bishop] & to)
		c = bishop;
	else if (ppieces[movenotof][knight] & to)
		c = knight;
	else if (ppieces[movenotof][pawn] & to)
		c = pawn;

	// if castling move, move the rook
	if (p == king) {
		if (m.move == wkoo[moveof]) {
			pall[moveof] ^= wroo[moveof];
			ppieces[moveof][rook] ^= wroo[moveof];
		} else if (m.move == wkooo[moveof]) {
			pall[moveof] ^= wrooo[moveof];
			ppieces[moveof][rook] ^= wrooo[moveof];
		}
		if (!kingmovedat[moveof])
			kingmovedat[moveof] = movenumber + 1;
	} else if (p == rook) {
		int side = (from & fileh) == 0;
		if (!rookmovedat[moveof][side])
			rookmovedat[moveof][side] = movenumber + 1;
	} else if (p == pawn) {
		// set en passent info
		if ((from & rank2) && (to & rank4)) {
			enpasqr[movenumber + 1] = to >> 8;
			enpapwn[movenumber + 1] = to;
		} else if ((from & rank7) && (to & rank5)) {
			enpasqr[movenumber + 1] = to << 8;
			enpapwn[movenumber + 1] = to;
		}
	}
	// move individual pieces
	if (p == pawn) {
		if (to & (rank8 | rank1)) {
			ppieces[moveof][pawn] ^= from;
			ppieces[moveof][m.promto] |= to;
		} else
			ppieces[moveof][p] ^= m.move;
	}

// en passent and opponents piece removal
	if (c) {
		pall[movenotof] ^= to;
		ppieces[movenotof][c] ^= to;
	} else if (to & enpasqr[movenumber]) {
		pall[movenotof] ^= enpapwn[movenumber];
		ppieces[movenotof][pawn] ^= enpapwn[movenumber];
	}

	history[movenumber].copy(m);
	captured[movenumber] = c;

	movenumber++;

	moveof = movenotof;
	movenotof = moveof == white ? black : white;
}

void board::undolastmove() {
}
