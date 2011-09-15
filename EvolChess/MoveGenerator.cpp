/*
 * MoveGenerator.cpp
 *
 *  Created on: Sep 11, 2011
 *      Author: baole
 */

#include "MoveGenerator.h"

MoveGenerator::MoveGenerator() {
	// TODO Auto-generated constructor stub

}

MoveGenerator::~MoveGenerator() {
	// TODO Auto-generated destructor stub
}

// get position of a bit in integer
int MoveGenerator::get_bit_pos(bitboard b) {
	int pos = -1;
	while (b) {
		b >>= 1;
		pos += 1;
	}
	return pos;
}

void MoveGenerator::gen_king_atk(board *b, bitboard & atkbrd)
{
	//get king position
		bitboard lsb = b->ppieces[b->moveof][king];
		int lsbint = get_bit_pos(lsb);
	// get all moves on that position
		atkbrd |= king_moves[lsbint];
}

void MoveGenerator::gen_pawn_atk(board *b, bitboard & atkbrd)
{
	//iterate thru' all the pawns
		bitboard ap = b->ppieces[b->moveof][pawn]; //all pawns
		bitboard lsb; //last significant bit
		bitboard c; //moved to position, captured to position

		while (ap) {
			/*this is a good code to get the last significant bit.
			 works like this:
			 ap				= 00101100
			 ~ap				= 11010011
			 ~ap + 1			= 11010100
			 ap & (~ap + 1)	= 00000100
			 */
			lsb = ap & (~ap + 1);
			//if they are white
			if (b->moveof == white) {
				//lets checkout capture squares
				//capturing on right?
				c = lsb << 9;
				//make sure it doesnt hit the wall
				c &= ~filea;
				//make sure it captures adversary only
				c &= b->pall[black];
				//record capture
				atkbrd |= c;
				//capture on left!
				c = lsb << 7;
				c &= ~fileh;
				c &= b->pall[black];
				atkbrd |= c;
			}
			//same goes for the black pawns
			//just for the fact that they move in opposite direction.
			else {
				//capture moves for black
				c = lsb >> 9;
				c &= ~fileh;
				c &= b->pall[white];
				atkbrd |= c;

				c = lsb >> 7;
				c &= ~filea;
				c &= b->pall[white];
				atkbrd |= c;
			}
			//remove the lsb from pawns bits
			ap ^= lsb;
		}
}

void MoveGenerator::gen_knight_atk(board *b, bitboard & atkbrd)
{
	//get position
		bitboard an = b->ppieces[b->moveof][knight];
		bitboard lsb; //last significant bit
		int lsbint;
		bitboard ant/*all move to positions*/;
	//iterate thru' all knights
		while (an) {
			lsb = an & (~an + 1);
			lsbint = get_bit_pos(lsb);
			// get all knight moves on that position
			ant = knight_moves[lsbint];
			// loop thru' all moves
			atkbrd |= ant;
			//remove the lsb from pawns bits
			an ^= lsb;
		}
}

void MoveGenerator::gen_rook_atk(board *b, bitboard & atkbrd)
{
	// squares not occupied by our pieces
		bitboard othersq = ~b->pall[b->moveof];
	// squares occupied by all pieces
		bitboard _all = b->pall[white] | b->pall[black];

		bitboard lsb; //last significant bit
		int lsbint;
		bitboard _rm, _lm, _um, _dm;
	//iterate thru' all rooks/queens
		bitboard ar = b->ppieces[b->moveof][rook] | b->ppieces[b->moveof][queen];
		while (ar) {
			lsb = ar & (~ar + 1);
			lsbint = get_bit_pos(lsb);
			//generate moves to the right
			_rm = right_moves[lsbint] & _all;
			_rm = _rm << 1 | _rm << 2 | _rm << 3 | _rm << 4 | _rm << 5 | _rm << 6;
			_rm &= right_moves[lsbint];
			_rm ^= right_moves[lsbint];
			_rm &= othersq;

			// generate moves to the left
			_lm = left_moves[lsbint] & _all;
			_lm = _lm >> 1 | _lm >> 2 | _lm >> 3 | _lm >> 4 | _lm >> 5 | _lm >> 6;
			_lm &= left_moves[lsbint];
			_lm ^= left_moves[lsbint];
			_lm &= othersq;

			// generate moves to the top
			_um = up_moves[lsbint] & _all;
			_um = _um << 8 | _um << 16 | _um << 24 | _um << 32 | _um << 40
					| _um << 48;
			_um &= up_moves[lsbint];
			_um ^= up_moves[lsbint];
			_um &= othersq;

			// generate moves to the bottom
			_dm = down_moves[lsbint] & _all;
			_dm = _dm >> 8 | _dm >> 16 | _dm >> 24 | _dm >> 32 | _dm >> 40
					| _dm >> 48;
			_dm &= down_moves[lsbint];
			_dm ^= down_moves[lsbint];
			_dm &= othersq;

			// loop thru' all moves
			_dm = _dm | _um | _lm | _rm;
			atkbrd |= _dm;
			ar ^= lsb;
		}
}

void MoveGenerator::gen_bishop_atk(board *b, bitboard & atkbrd)
{
	// squares not occupied by our pieces
		bitboard othersq = ~b->pall[b->moveof];
	// squares occupied by all pieces
		bitboard _all = b->pall[white] | b->pall[black];

		bitboard lsb; //last significant bit
		int lsbint;
		bitboard _45m, _225m, _135m, _315m;
	//iterate thru' all bishops/queens
		bitboard ab = b->ppieces[b->moveof][bishop] | b->ppieces[b->moveof][queen];
		while (ab) {
			lsb = ab & (~ab + 1);
			lsbint = get_bit_pos(lsb);
			//generate moves for diagonally right up
			_45m = deg45_moves[lsbint] & _all;
			_45m = _45m << 9 | _45m << 18 | _45m << 27 | _45m << 36 | _45m << 45
					| _45m << 54;
			_45m &= deg45_moves[lsbint];
			_45m ^= deg45_moves[lsbint];
			_45m &= othersq;

			// generate moves for left down
			_225m = deg225_moves[lsbint] & _all;
			_225m = _225m >> 9 | _225m >> 18 | _225m >> 27 | _225m >> 36
					| _225m >> 45 | _225m >> 54;
			_225m &= deg225_moves[lsbint];
			_225m ^= deg225_moves[lsbint];
			_225m &= othersq;

			// generate moves right down
			_135m = deg135_moves[lsbint] & _all;
			_135m = _135m >> 7 | _135m >> 14 | _135m >> 21 | _135m >> 28
					| _135m >> 35 | _135m >> 42;
			_135m &= deg135_moves[lsbint];
			_135m ^= deg135_moves[lsbint];
			_135m &= othersq;

			// generate moves for left up
			_315m = deg315_moves[lsbint] & _all;
			_315m = _315m << 7 | _315m << 14 | _315m << 21 | _315m << 28
					| _315m << 35 | _315m << 42;
			_315m &= deg315_moves[lsbint];
			_315m ^= deg315_moves[lsbint];
			_315m &= othersq;

			// loop thru' all moves
			_315m = _315m | _135m | _225m | _45m;
			atkbrd |= _315m;
			ab ^= lsb;
		}
}

void MoveGenerator::generate(board & b, vector<bitmove*> & v) {
	gen_king_moves(b, v);
	//gen_queen_moves(b, v); // combined with rook and bishop moves
	gen_rook_moves(b, v);
	gen_bishop_moves(b, v);
	gen_knight_moves(b, v);
	gen_pawn_moves(b, v);
}

void MoveGenerator::gen_king_moves(board & b, vector<bitmove*> & v) {
	side moveof = b.moveof;
	// squares not occupied by our pieces
	bitboard othersq = ~b.pall[moveof], _all = b.pall[white] | b.pall[black];
	//get king position
	bitboard lsb = b.ppieces[moveof][king];
	bitboard m; //moved to position
	int lsbint;
	bitboard ant/*all move to positions*/;
	lsbint = get_bit_pos(lsb);
	// get all moves on that position
	ant = king_moves[lsbint];

	// generate castling move if allowed
	if (!b.kingmovedat[moveof]) {
		// castle towards filea
		if (!b.rookmovedat[moveof][0]) {
			if (moveof == white && !(_all & (b1 | c1 | d1))
					&& (b.ppieces[white][rook] & a1)) {
				v.push_back(new bitmove(lsb, c1));
			} else if (moveof == black && !(_all & (b8 | c8 | d8))
					&& (b.ppieces[black][rook] & a8)) {
				v.push_back(new bitmove(lsb, c8));
			}
		}
		// castle towards fileh
		if (!b.rookmovedat[moveof][1]) {
			if (moveof == white && !(_all & (f1 | g1))
					&& (b.ppieces[white][rook] & h1)) {
				v.push_back(new bitmove(lsb, g1));
			} else if (moveof == black && !(_all & (f8 | g8))
					&& (b.ppieces[black][rook] & h8)) {
				v.push_back(new bitmove(lsb, g1));
			}
		}
	}
	// loop thru' all moves
	while (ant) {
		m = ant & (~ant + 1);
		if (m & othersq) {
			v.push_back(new bitmove(lsb, m));
		}
		ant ^= m;
	}
}

void MoveGenerator::gen_queen_moves(board & b, vector<bitmove*> & v) {
}

void MoveGenerator::gen_rook_moves(board & b, vector<bitmove*> & v) {
	side moveof = b.moveof;
	//get position
	bitboard ar = b.ppieces[moveof][rook];
	ar |= b.ppieces[moveof][queen];
// squares not occupied by our pieces
	bitboard othersq = ~b.pall[moveof];
// squares occupied by all pieces
	bitboard _all = b.pall[white] | b.pall[black];

	bitboard lsb; //last significant bit
	bitboard m; //moved to position
	int lsbint;
	bitboard _rm, _lm, _um, _dm;
//iterate thru' all rooks
	while (ar) {
		lsb = ar & (~ar + 1);
		lsbint = get_bit_pos(lsb);
		//generate moves to the right
		_rm = right_moves[lsbint] & _all;
		_rm = _rm << 1 | _rm << 2 | _rm << 3 | _rm << 4 | _rm << 5 | _rm << 6;
		_rm &= right_moves[lsbint];
		_rm ^= right_moves[lsbint];
		_rm &= othersq;

		// generate moves to the left
		_lm = left_moves[lsbint] & _all;
		_lm = _lm >> 1 | _lm >> 2 | _lm >> 3 | _lm >> 4 | _lm >> 5 | _lm >> 6;
		_lm &= left_moves[lsbint];
		_lm ^= left_moves[lsbint];
		_lm &= othersq;

		// generate moves to the top
		_um = up_moves[lsbint] & _all;
		_um = _um << 8 | _um << 16 | _um << 24 | _um << 32 | _um << 40
				| _um << 48;
		_um &= up_moves[lsbint];
		_um ^= up_moves[lsbint];
		_um &= othersq;

		// generate moves to the bottom
		_dm = down_moves[lsbint] & _all;
		_dm = _dm >> 8 | _dm >> 16 | _dm >> 24 | _dm >> 32 | _dm >> 40
				| _dm >> 48;
		_dm &= down_moves[lsbint];
		_dm ^= down_moves[lsbint];
		_dm &= othersq;

		// loop thru' all moves
		_dm = _dm | _um | _lm | _rm;
		while (_dm) {
			m = _dm & (~_dm + 1);
			if (m & othersq) {
				v.push_back(new bitmove(lsb, m, none));
			}
			_dm ^= m;
		}
		ar ^= lsb;
	}
}

void MoveGenerator::gen_bishop_moves(board & b, vector<bitmove*> & v) {
	side moveof = b.moveof;
	//get position
	bitboard ab = b.ppieces[moveof][bishop];
	ab |= b.ppieces[moveof][queen];
	// squares not occupied by our pieces
	bitboard othersq = ~b.pall[moveof];
	// squares occupied by all pieces
	bitboard _all = b.pall[white] | b.pall[black];

	bitboard lsb; //last significant bit
	bitboard m; //moved to position
	int lsbint;
	bitboard _45m, _225m, _135m, _315m;
	//iterate thru' all bishops
	while (ab) {
		lsb = ab & (~ab + 1);
		lsbint = get_bit_pos(lsb);
		//generate moves for diagonally right up
		_45m = deg45_moves[lsbint] & _all;
		_45m = _45m << 9 | _45m << 18 | _45m << 27 | _45m << 36 | _45m << 45
				| _45m << 54;
		_45m &= deg45_moves[lsbint];
		_45m ^= deg45_moves[lsbint];
		_45m &= othersq;

		// generate moves for left down
		_225m = deg225_moves[lsbint] & _all;
		_225m = _225m >> 9 | _225m >> 18 | _225m >> 27 | _225m >> 36
				| _225m >> 45 | _225m >> 54;
		_225m &= deg225_moves[lsbint];
		_225m ^= deg225_moves[lsbint];
		_225m &= othersq;

		// generate moves right down
		_135m = deg135_moves[lsbint] & _all;
		_135m = _135m >> 7 | _135m >> 14 | _135m >> 21 | _135m >> 28
				| _135m >> 35 | _135m >> 42;
		_135m &= deg135_moves[lsbint];
		_135m ^= deg135_moves[lsbint];
		_135m &= othersq;

		// generate moves for left up
		_315m = deg315_moves[lsbint] & _all;
		_315m = _315m << 7 | _315m << 14 | _315m << 21 | _315m << 28
				| _315m << 35 | _315m << 42;
		_315m &= deg315_moves[lsbint];
		_315m ^= deg315_moves[lsbint];
		_315m &= othersq;

		// loop thru' all moves
		_315m = _315m | _135m | _225m | _45m;
		while (_315m) {
			m = _315m & (~_315m + 1);
			v.push_back(new bitmove(lsb, m, none));
			_315m ^= m;
		}
		ab ^= lsb;
	}
}

// generate knight moves
void MoveGenerator::gen_knight_moves(board &b, vector<bitmove*> &v) {
	side moveof = b.moveof;

// squares not occupied by our pieces
	bitboard othersq = ~b.pall[moveof];
//get position
	bitboard an = b.ppieces[moveof][knight];
	bitboard lsb; //last significant bit
	bitboard m; //moved to position
	int lsbint;
	bitboard ant/*all move to positions*/;
//iterate thru' all knights
	while (an) {
		lsb = an & (~an + 1);
		lsbint = get_bit_pos(lsb);
		// get all knight moves on that position
		ant = knight_moves[lsbint];
		// loop thru' all moves
		while (ant) {
			m = ant & (~ant + 1);
			if (m & othersq)
				v.push_back(new bitmove(lsb, m, none));
			ant ^= m;
		}
		//remove the lsb from pawns bits
		an ^= lsb;
	}
}

void MoveGenerator::gen_pawn_moves(board & b, vector<bitmove*> & v) {
	side moveof = b.moveof;
	// squares not occupied by any piece
	bitboard emptysq = ~(b.pall[white] | b.pall[black]);
	//iterate thru' all the pawns
	bitboard ap = b.ppieces[moveof][pawn]; //all pawns
	bitboard lsb; //last significant bit
	bitboard m, c; //moved to position, captured to position

	while (ap) {
		/*this is a good code to get the last significant bit.
		 works like this:
		 ap				= 00101100
		 ~ap				= 11010011
		 ~ap + 1			= 11010100
		 ap & (~ap + 1)	= 00000100
		 */
		lsb = ap & (~ap + 1);
		//if they are white
		if (moveof == white) {
			//this gives one rank forward
			m = lsb << 8;
			//as pawn moves on empty squares
			if (m & emptysq) {
				//found move now push it on the stack
				//also check if the pawn did manage to reach its final frontier
				if (m & rank8) //pawn is promoted to...
						{
					v.push_back(new bitmove(lsb, m, queen));
					v.push_back(new bitmove(lsb, m, rook));
					v.push_back(new bitmove(lsb, m, bishop));
					v.push_back(new bitmove(lsb, m, knight));
				} else
					v.push_back(new bitmove(lsb, m, none));
				//now for the double move
				//only pawns at rank 2 can do that
				if (lsb & rank2) {
					//one more rank forward
					m <<= 8;
					if (m & emptysq)
						//push it in...
						//this is a double move, flag it also
						v.push_back(new bitmove(lsb, m, none));
				}
			}
			//lets checkout capture squares
			//capturing on right?
			c = lsb << 9;
			//make sure it doesnt hit the wall
			c &= ~filea;
			//is this en passant
			if (c & b.enpasqr[b.movenumber])
				v.push_back(new bitmove(lsb, c, none));
			//make sure it captures adversary only
			c &= b.pall[black];
			//if capture exists then push it
			if (c) {
				if (c & rank8) //pawn is promoted to...
						{
					v.push_back(new bitmove(lsb, c, queen));
					v.push_back(new bitmove(lsb, c, rook));
					v.push_back(new bitmove(lsb, c, bishop));
					v.push_back(new bitmove(lsb, c, knight));
					} else
						v.push_back(new bitmove(lsb, c, none));
			}
			//capture on left!
			c = lsb << 7;
			c &= ~fileh;
			if (c & b.enpasqr[b.movenumber])
				v.push_back(new bitmove(lsb, c, none));
			c &= b.pall[black];
			if (c) {
				if (c & rank8) //pawn is promoted to...
						{
					v.push_back(new bitmove(lsb, c, queen));
					v.push_back(new bitmove(lsb, c, rook));
					v.push_back(new bitmove(lsb, c, bishop));
					v.push_back(new bitmove(lsb, c, knight));
					} else
						v.push_back(new bitmove(lsb, c, none));
			}

		}
		//same goes for the black pawns
		//just for the fact that they move in opposite direction.
		else {
			//one rank forward for black pawn in different direction
			m = lsb >> 8;
			if (m & emptysq) {
				if (m & rank1) //pawn is promoted to...
						{
					v.push_back(new bitmove(lsb, m, queen));
					v.push_back(new bitmove(lsb, m, rook));
					v.push_back(new bitmove(lsb, m, bishop));
					v.push_back(new bitmove(lsb, m, knight));
				} else
					v.push_back(new bitmove(lsb, m, none));
				//double move! for black it will be from rank 7.
				if (lsb & rank7) {
					m >>= 8;
					if (m & emptysq)
						v.push_back(new bitmove(lsb, m, none));
				}
			}
			//capture moves for black
			c = lsb >> 9;
			c &= ~fileh;
			if (c & b.enpasqr[b.movenumber])
				v.push_back(new bitmove(lsb, c, none));
			c &= b.pall[white];
			if (c) {
				if (c & rank1) //pawn is promoted to...
						{
					v.push_back(new bitmove(lsb, c, queen));
					v.push_back(new bitmove(lsb, c, rook));
					v.push_back(new bitmove(lsb, c, bishop));
					v.push_back(new bitmove(lsb, c, knight));
					} else
						v.push_back(new bitmove(lsb, c, none));
			}
			c = lsb >> 7;
			c &= ~filea;
			if (c & b.enpasqr[b.movenumber])
				v.push_back(new bitmove(lsb, c, none));
			c &= b.pall[white];
			if (c) {
				if (c & rank1) //pawn is promoted to...
						{
					v.push_back(new bitmove(lsb, c, queen));
					v.push_back(new bitmove(lsb, c, rook));
					v.push_back(new bitmove(lsb, c, bishop));
					v.push_back(new bitmove(lsb, c, knight));
					} else
						v.push_back(new bitmove(lsb, c, none));
			}
		}
		//remove the lsb from pawns bits
		ap ^= lsb;
	}
}

