/*
 * board.cpp
 *
 *  Created on: Sep 4, 2011
 *      Author: baole
 */

#include "board.h"

board::board() {
	isready = 0;
}

void board::newgame() {
	// initialize to a standard new game
	movenumber = 0;
	moveof = white;
	movenotof = black;

	kingmovedat[white] = 0;
	kingmovedat[black] = 0;

	rookmovedat[white][0] = 0;
	rookmovedat[white][1] = 0;
	rookmovedat[black][0] = 0;
	rookmovedat[black][1] = 0;

	enpasqr[0] = 0;
	enpapwn[0] = 0;

	//place all white and black pieces
	for (int s = 0; s < 2; s++) {
		pall[s] = start_all[s];
		for (int p = 0; p < 6; p++)
			ppieces[s][p] = start_pieces[s][p];
	}

	isready = 1;
}

void board::domove(const bitmove &m) {
	if (!isready) {
		cout << "board not ready!\n";
		return;
	}
	bitboard from = m.from;
	bitboard to = m.to;
	bitboard mov = from | to;

	// adjust all pieces bitboard
	pall[moveof] ^= mov;

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
	moved[movenumber] = p;

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
	captured[movenumber] = c;

	// if castling move, move the rook
	if (p == king) {
		if (mov == wkoo[moveof]) {
			pall[moveof] ^= wroo[moveof];
			ppieces[moveof][rook] ^= wroo[moveof];
		} else if (mov == wkooo[moveof]) {
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
		// set en passent info for double moves
		if ((from & rank2) && (to & rank4)) {
			enpasqr[movenumber + 1] = to >> 8;
			enpapwn[movenumber + 1] = to;
		} else if ((from & rank7) && (to & rank5)) {
			enpasqr[movenumber + 1] = to << 8;
			enpapwn[movenumber + 1] = to;
		} else {
			enpasqr[movenumber + 1] = 0;
			enpapwn[movenumber + 1] = 0;
		}
	}
	// move individual pieces
	if (p == pawn && (to & (rank8 | rank1))) {
		ppieces[moveof][pawn] ^= from;
		ppieces[moveof][m.promto] |= to;
	} else
		ppieces[moveof][p] ^= mov;

// en passent and opponents piece removal
	if (to & enpasqr[movenumber]) {
		pall[movenotof] ^= enpapwn[movenumber];
		ppieces[movenotof][pawn] ^= enpapwn[movenumber];
	} else if (c != none) {
		pall[movenotof] ^= to;
		ppieces[movenotof][c] ^= to;
	}

	history[movenumber].copy(m);

	movenumber++;

	moveof = movenotof;
	movenotof = moveof == white ? black : white;
}

void board::undolastmove() {
	if (!isready) {
		cout << "board not ready!\n";
		return;
	}
	movenumber--;
	moveof = movenotof;
	movenotof = moveof == white ? black : white;
	bitmove m(history[movenumber]);

	bitboard from = m.from;
	bitboard to = m.to;
	bitboard mov = from | to;

	// get the pieces involved
	piece p = none;
	if (ppieces[moveof][king] & to)
		p = king;
	else if (ppieces[moveof][queen] & to)
		p = queen;
	else if (ppieces[moveof][rook] & to)
		p = rook;
	else if (ppieces[moveof][bishop] & to)
		p = bishop;
	else if (ppieces[moveof][knight] & to)
		p = knight;
	else if (ppieces[moveof][pawn] & to)
		p = pawn;

	piece c = captured[movenumber];

	// en passent and opponents piece removal
	if (c != none) {
		pall[movenotof] ^= to;
		ppieces[movenotof][c] ^= to;
	} else if (to & enpasqr[movenumber]) {
		pall[movenotof] ^= enpapwn[movenumber];
		ppieces[movenotof][pawn] ^= enpapwn[movenumber];
	}
	// move individual pieces
	if (m.promto!=none) {
		ppieces[moveof][pawn] ^= from;
		ppieces[moveof][m.promto] ^= to;
	} else
		ppieces[moveof][p] ^= mov;

	// adjust all pieces bitboard
	pall[moveof] ^= mov;

	// if castling move, move the rook also
	if (p == king) {
		if (mov == wkoo[moveof]) {
			pall[moveof] ^= wroo[moveof];
			ppieces[moveof][rook] ^= wroo[moveof];
		} else if (mov == wkooo[moveof]) {
			pall[moveof] ^= wrooo[moveof];
			ppieces[moveof][rook] ^= wrooo[moveof];
		}
		//if (kingmovedat[moveof])
		kingmovedat[moveof] = 0;
	} else if (p == rook) {
		int side = (from & fileh) == 0;
		//if (rookmovedat[moveof][side])
		rookmovedat[moveof][side] = 0;
	} else if (p == pawn) {
		// set en passent info
		if ((from & rank2) && (to & rank4)) {
			enpasqr[movenumber + 1] = 0;
			enpapwn[movenumber + 1] = 0;
		} else if ((from & rank7) && (to & rank5)) {
			enpasqr[movenumber + 1] = 0;
			enpapwn[movenumber + 1] = 0;
		}
	}
}

ostream &operator<<(ostream &s, board m) {
	if (!m.isready) {
		cout << "board not ready!\n";
		return s;
	}
	s << "\n";
	bitboard p = 0;
	char toprint[2];

	for (int r = 8; r > 0; r--) {
		cout << r << " ";
		for (int f = 0; f < 8; f++) {
			p = rank[r - 1] & file[f];
			strcpy(toprint, "-");
			for (int i = 0; i < 6; i++)
				if (m.ppieces[white][i] & p) {
					strcpy(toprint, notationw[i]);
					break;
				} else if (m.ppieces[black][i] & p) {
					strcpy(toprint, notationb[i]);
					break;
				}
			s << toprint << " ";
		}
		s << endl;
	}
	s << "  a b c d e f g h";
	s << endl;

	return s;
}
