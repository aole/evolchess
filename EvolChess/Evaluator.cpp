/*
 * Evaluator.cpp
 *
 *  Created on: Sep 11, 2011
 *      Author: Bhupendra Aole
 */

#include "Evaluator.h"



int Evaluator::score(board & b)
{
	//calculate board value
		int bv[] = { 0, 0 }, j;
		bitboard lsb; //last significant bit

		for (int i = 0; i < 2; i++) {
			if (i == 0)
				j = 1;
			else
				j = 1;

			//piece value
			bitboard ap = b.pall[i];
			while (ap) {
				lsb = ap & (~ap + 1);
				for (int j = 0; j < 6; j++) {
					if (lsb & b.ppieces[i][j]) {
						bv[i] += piecevalue[j];
						break;
					}
				}
				ap ^= lsb;
			}
			//check for center pawns
			if (b.ppieces[i][pawn] & 0x1818000000ULL)
				bv[i] += 2;
			/*if (pieces[i][pawn] & 0x181818180000ULL)
			 bv[i]++;*/

			// check for castling
			if (i == white) {
				if ((b.ppieces[i][king] & 0x40) && !(b.pall[i] & 0x80)
						&& (b.ppieces[i][pawn] & 0xE000)) {
					bv[i] += 10;
				}
				//pawn advancement
				if (b.ppieces[i][pawn] & rank7)
					bv[i] += 3;
				if (b.ppieces[i][pawn] & rank6)
					bv[i] += 3;
				if (b.ppieces[i][pawn] & rank5)
					bv[i] += 3;
				if (b.ppieces[i][pawn] & rank4)
					bv[i] += 2;
				if (b.ppieces[i][pawn] & rank3)
					bv[i] += 1;

				// blocking central pawns
				if ((b.ppieces[i][pawn] & d2) && ((b.pall[white] | b.pall[black]) & d3))
					bv[i] -= 3;
				if ((b.ppieces[i][pawn] & e2) && ((b.pall[white] | b.pall[black]) & e3))
					bv[i] -= 3;
			} else {
				if ((b.ppieces[i][king] & 0x4000000000000000ULL)
						&& !(b.pall[i] & 0x8000000000000000ULL)
						&& (b.ppieces[i][pawn] & 0xE000000000000000ULL)) {
					bv[i] += 10;
				}
				//pawn advancement
				if (b.ppieces[i][pawn] & rank2)
					bv[i] += 3;
				if (b.ppieces[i][pawn] & rank3)
					bv[i] += 3;
				if (b.ppieces[i][pawn] & rank4)
					bv[i] += 3;
				if (b.ppieces[i][pawn] & rank5)
					bv[i] += 2;
				if (b.ppieces[i][pawn] & rank6)
					bv[i] += 1;

				// blocking central pawns
				if ((b.ppieces[i][pawn] & d7) && ((b.pall[white] | b.pall[black]) & d6))
					bv[i] -= 3;
				if ((b.ppieces[i][pawn] & e7) && ((b.pall[white] | b.pall[black]) & e6))
					bv[i] -= 3;
			}
			//knights in the middle
			if (b.ppieces[i][knight] & (d4 | d5 | e4 | e5))
				bv[i] += 4;
			if (b.ppieces[i][knight]
					& (c3 | d3 | e3 | f3 | c6 | d6 | e6 | f6 | c4 | f4))
				bv[i] += 3;
			//bishops on principle diagonals
			if (b.ppieces[i][bishop] & (0x8040201008040201ULL | 0x0804020180402010ULL))
				bv[i] += 3;
			if (b.ppieces[i][bishop] & (0xC0E070381C0E0703ULL | 0x03070E1C3870E0C0ULL))
				bv[i] += 3;
			//rook on open file
			if ((b.ppieces[i][rook] & filea) && !(b.ppieces[i][pawn] & filea))
				bv[i] += 4;
			if ((b.ppieces[i][rook] & fileb) && !(b.ppieces[i][pawn] & fileb))
				bv[i] += 4;
			if ((b.ppieces[i][rook] & filec) && !(b.ppieces[i][pawn] & filec))
				bv[i] += 4;
			if ((b.ppieces[i][rook] & filed) && !(b.ppieces[i][pawn] & filed))
				bv[i] += 4;
			if ((b.ppieces[i][rook] & filee) && !(b.ppieces[i][pawn] & filee))
				bv[i] += 4;
			if ((b.ppieces[i][rook] & filef) && !(b.ppieces[i][pawn] & filef))
				bv[i] += 4;
			if ((b.ppieces[i][rook] & fileg) && !(b.ppieces[i][pawn] & fileg))
				bv[i] += 4;
			if ((b.ppieces[i][rook] & fileh) && !(b.ppieces[i][pawn] & fileh))
				bv[i] += 4;
		}
	//return bv[movenotof] - bv[moveof];
		return bv[b.moveof] - bv[b.movenotof];
}

