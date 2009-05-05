/* * * * * * * * * *
 * EvolChess
 * * * * * * * * * *
 * engine.cpp
 *
 *  Created on: May 1, 2009
 *      Author: Bhupendra Aole
 */

#include <iostream>

#include "engine.h"

using namespace std;

void cmovestack::push(bitboard f, bitboard t, byte moved, byte captured = 0, byte promto = 0, byte flags = 0, bitboard mov2 = 0) {
    allMoves |= t;
    move[top].from = f;
    move[top].to = t;
    move[top].mov2 = mov2;
    move[top].piece = moved;
    move[top].promotedto = promto;
    move[top].flags = flags;
    //get the captured piece
    move[top].captured = captured;

    top++;
}

int cmovestack::find (bitboard f, bitboard t, int promto = 0) {
	for (int i=0; i<top; i++)
	//operation '&' should do here as all these(variables)
	//will have only one bit activated.
	if (move[i].from == f && move[i].to == t && move[i].promotedto == promto)
		return i;
	//if we didnt find the move then return '-1' representing error for us.
	return -1;
}

char *cmovestack::getMoveTxt(int index) {
    cmove m = move[index];

    if (m.from & filea)
       mov[0] = 'a';
    else if (m.from & fileb)
       mov[0] = 'b';
    else if (m.from & filec)
       mov[0] = 'c';
    else if (m.from & filed)
       mov[0] = 'd';
    else if (m.from & filee)
       mov[0] = 'e';
    else if (m.from & filef)
       mov[0] = 'f';
    else if (m.from & fileg)
       mov[0] = 'g';
    else if (m.from & fileh)
       mov[0] = 'h';

    if (m.from & rank1)
       mov[1] = '1';
    else if (m.from & rank2)
       mov[1] = '2';
    else if (m.from & rank3)
       mov[1] = '3';
    else if (m.from & rank4)
       mov[1] = '4';
    else if (m.from & rank5)
       mov[1] = '5';
    else if (m.from & rank6)
       mov[1] = '6';
    else if (m.from & rank7)
       mov[1] = '7';
    else if (m.from & rank8)
       mov[1] = '8';

    if (m.to & filea)
       mov[2] = 'a';
    else if (m.to & fileb)
       mov[2] = 'b';
    else if (m.to & filec)
       mov[2] = 'c';
    else if (m.to & filed)
       mov[2] = 'd';
    else if (m.to & filee)
       mov[2] = 'e';
    else if (m.to & filef)
       mov[2] = 'f';
    else if (m.to & fileg)
       mov[2] = 'g';
    else if (m.to & fileh)
       mov[2] = 'h';

    if (m.to & rank1)
       mov[3] = '1';
    else if (m.to & rank2)
       mov[3] = '2';
    else if (m.to & rank3)
       mov[3] = '3';
    else if (m.to & rank4)
       mov[3] = '4';
    else if (m.to & rank5)
       mov[3] = '5';
    else if (m.to & rank6)
       mov[3] = '6';
    else if (m.to & rank7)
       mov[3] = '7';
    else if (m.to & rank8)
       mov[3] = '8';

    if (m.promotedto < 5 && m.promotedto > 0) {
       mov[4] = notationb[m.promotedto][0];
       mov[5] = '\0';
    }
    else
       mov[4] = '\0';

    return mov;
}

Engine::Engine() {}

Engine::~Engine() {}

//initializes the game to starting position
void Engine::newGame() {
	stack.init();
	moveshistory.init();
	//place all white and black pieces
	for (int s=0; s<2; s++)
	{
		all[s] = start_all[s];
		for (int p=0;p<6; p++)
			pieces[s][p] = start_pieces[s][p];
	}
	//move of white
	moveof = white;
	movenotof = black;

	kingmoved[0] = kingmoved[1] = 0;
	rookmoved[0][0] = rookmoved[0][1] = rookmoved[1][0] = rookmoved[1][1] = 0;
}

//show the board on the console
//empty squares are shown by '*'
void Engine::show_board()
{
	cout<<"\n";
	bitboard p = 0;
	char toprint[2];

	for (int r=8;r>0;r--)
	{
		cout<<r<<" ";
		for (int f=0;f<8;f++)
		{
            p = rank[r-1] & file[f];
			strcpy(toprint, "*");
			for (int i=0;i<6;i++)
				if (pieces[white][i] & p)
				{
					strcpy(toprint, notationw[i]);
					break;
				}
			for (int i=0;i<6;i++)
				if (pieces[black][i] & p)
				{
					strcpy(toprint, notationb[i]);
					break;
				}
			cout<<toprint<<" ";
		}
		cout<<endl;
	}
	cout<<"  ";
	for (char f='a';f<='h';f++)
		cout<<f<<" ";
	cout<<endl;
}
//undo last move
void Engine::undolastmove() {
	cmove m = moveshistory.pop();
	//switch move of
	moveof = movenotof;
	movenotof = (moveof==white?black:white);
	//undo move
	all[moveof] ^= (m.from | m.to);
	pieces[moveof][m.piece] ^= (m.from | m.to);
	//undo II move if there
	if (m.mov2) {
		all[moveof] ^= m.mov2;
		pieces[moveof][rook] ^= m.mov2;
	}
	//respawn captured piece
	if (m.captured) {
		all[movenotof] |= m.to;
		pieces[movenotof][m.captured] |= m.to;
	}
}
//
int Engine::doaimove() {
	int rndm;
    // make a random move
    rndm = rand() % stack.top;
    return domove(rndm);
}
//change the board according to the move
//we get the move from the index to the moves stack.
int Engine::domove(int index)
{
    if (index < 0)
       return index;

    cmove move = stack.move[index];

	bitboard f = move.from;
	bitboard t = move.to;
	bitboard mov = f | t, mov2 = move.mov2;

	byte pt = move.promotedto;

    //check for castling moves
	if (mov2) {
		all[moveof] ^= mov2;
		pieces[moveof][rook] ^= mov2;
	}
	//move the pieces
	/*works like this:
	  0 0 0   0 1 0   0 1 0
	  1 1 1 ^ 0 1 0 = 1 0 1
	  1 1 1   0 0 0   1 1 1
	  [all] ^ [mov] =[result]
	*/
	// check if king was moved. cannot castle then
	if (f & pieces[moveof][king])
	    kingmoved[moveof] = 1;
    // check if rook was moved. in that case, cannot castle with that rook
    if (f & pieces[moveof][rook] && f & filea)
        rookmoved[moveof][0] = 1;
    else if (f & pieces[moveof][rook] && f & fileh)
        rookmoved[moveof][1] = 1;

	all[moveof] ^= mov;
	//check which piece was that and move it
	for (int i=0;i<6;i++)
		if (f & pieces[moveof][i])
		{
			if (pt)
			{
				pieces[moveof][i] ^= f;
				pieces[moveof][pt] |= t;
			}else
				pieces[moveof][i] ^= mov;
			break;
		}
	//if this is enpassent then we need to adjust
	//for the capture of pawn at the right place
	if (move.flags & FLAGENPASSANT)
	{
		if (moveof == white)
			t >>= 8;//the pawn to be captured is actually, one rank back
		else
			t <<= 8;
	}
	//check for capture
	if (t & all[movenotof])
	{
		//remove the captured piece
		all[movenotof] ^= t;
		for (int i=0;i<6;i++)
			if (t & pieces[movenotof][i])
			{
				pieces[movenotof][i] ^= t;
				break;
			}
	}

	//if this is a double move
	//then we set the en passant square
	if (move.flags & FLAGDOUBLEMOVE)
	{
		if (moveof == white)
			epsq = f << 8;//enpassant square for white
		else
			epsq = f >> 8;//for black
	}else
		epsq = 0;//make sure to do this; we dont want any trouble in generate_move()

	//change the sides
	moveof = movenotof;
	movenotof = (moveof==white?black:white);

	//record the move
	moveshistory.push(move);

	return index;
}

//input that we got from the user
int Engine::input_move(char *m)
{
	bitboard f=1, t=1;
	//initially we suppose its not a promotion move:
	//cause its saved that way in cmove class
	byte pt=0;

	//check if string is legal
	/*m[0] and m[2] can be a, b, c, d, e, f, g, h
	  m[1] and m[3] can be 1, 2, 3, 4, 5, 6, 7, 8
	*/
	if (m[0] < 'a' || m[0] > 'h' ||
			m[1] < '1' || m[1] > '8' ||
			m[2] < 'a' || m[2] > 'h' ||
			m[3] < '1' || m[3] > '8')
		return -1;

	//convert into int
	int from = ((m[1] - '1')) * 8;
	from += m[0] - 'a';
	int to = ((m[3] - '1')) * 8;
	to += m[2] - 'a';

	//convert into bitboards
	f<<=from;
	t<<=to;

	//if promotion then 'promoted to' required.
	//m[4] can be q, r, b, n for obvious reasons
	if ((f & pieces[white][pawn] && moveof==white && t & rank8) ||
	    (f & pieces[black][pawn] && moveof==black && t & rank1))
	{
		if (strlen(m) < 5)
			return -1;
		if (m[4] == 'q')
			pt = queen;
		else if (m[4] == 'r')
			pt = rook;
		else if (m[4] == 'b')
			pt = bishop;
		else if (m[4] == 'n')
			pt = knight;
		else
			return -1;
	}

	//does move exists
	return stack.find (f, t, pt);
}
// get position of a bit in integer
int Engine::get_bit_pos(bitboard b) {
    int pos = -1;
    while (b) {
          b >>= 1;
          pos += 1;
    }
    return pos;
}
//push move to the moves stack
void Engine::push_move(bitboard f, bitboard t, byte moved, byte promto = 0, byte flags = 0, bitboard mov2 = 0){
	byte cap = 0;
	if (t & all[movenotof]){
		if (t & pieces[movenotof][pawn])
			cap=pawn;
		else if (t & pieces[movenotof][knight])
			cap=knight;
		else if (t & pieces[movenotof][bishop])
			cap=bishop;
		else if (t & pieces[movenotof][rook])
			cap=rook;
		else if (t & pieces[movenotof][queen])
			cap=queen;
	}
	stack.push (f, t, moved, cap, promto, flags, mov2);
}

//generate moves for a given pawn position
void Engine::gen_pawn_moves(side movefor) {
    // squares not occupied by any piece
	bitboard emptysq = ~(all[white] | all[black]);
	//iterate thru' all the pawns
	bitboard ap = pieces[movefor][pawn];//all pawns
	bitboard lsb;//last significant bit
	bitboard m, c;//moved to position, captured to position

	while (ap)
	{
		/*this is a good code to get the last significant bit.
		works like this:
		ap				= 00101100
		~ap				= 11010011
		~ap + 1			= 11010100
		ap & (~ap + 1)	= 00000100
		*/
		lsb = ap & (~ap + 1);
		//if they are white
		if (movefor==white)
		{
			//this gives one rank forward
			m = lsb << 8;
			//as pawn moves on empty squares
			if (m & emptysq)
			{
				//found move now push it on the stack
				//also check if the pawn did manage to reach its final frontier
				if (m & rank8)//pawn is promoted to...
				{
					push_move (lsb, m, pawn, queen);//queen
					push_move (lsb, m, pawn, rook);//rook
					push_move (lsb, m, pawn, bishop);//bishop
					push_move (lsb, m, pawn, knight);//knight
				}else
					push_move (lsb, m, pawn);
				//now for the double move
				//only pawns at rank 2 can do that
				if (lsb & rank2)
				{
					//one more rank forward
					m <<= 8;
					if (m & emptysq)
						//push it in...
						//this is a double move, flag it also
						push_move (lsb, m, pawn, 0, FLAGDOUBLEMOVE);
				}
			}
			//lets checkout capture squares
			//capturing on right?
			c = lsb << 9;
			//make sure it doesnt hit the wall
			c &= ~filea;
			//is this en passant
			if (c & epsq)
				stack.push (lsb, c, 0, FLAGENPASSANT);
			//make sure it captures adversary only
			c &= all[black];
			//if capture exists then push it
			if (c)
			{
				if (c & rank8)//pawn is promoted to...
				{
					push_move (lsb, c, pawn, queen);//queen
					push_move (lsb, c, pawn, rook);//rook
					push_move (lsb, c, pawn, bishop);//bishop
					push_move (lsb, c, pawn, knight);//knight
				}else
					push_move (lsb, c, pawn);
			}
			//capture on left!
			c = lsb << 7;
			c &= ~fileh;
			if (c & epsq)
				push_move (lsb, c, pawn, 0, FLAGENPASSANT);
			c &= all[black];
			if (c)
			{
				if (c & rank8)//pawn is promoted to...
				{
					push_move (lsb, c, pawn, queen);//queen
					push_move (lsb, c, pawn, rook);//rook
					push_move (lsb, c, pawn, bishop);//bishop
					push_move (lsb, c, pawn, knight);//knight
				}else
					push_move (lsb, c, pawn);
			}

		}
		//same goes for the black pawns
		//just for the fact that they move in opposite direction.
		else
		{
			//one rank forward for black pawn in different direction
			m = lsb >> 8;
			if (m & emptysq)
			{
				if (m & rank1)//pawn is promoted to...
				{
					push_move (lsb, m, pawn, queen);//queen
					push_move (lsb, m, pawn, rook);//rook
					push_move (lsb, m, pawn, bishop);//bishop
					push_move (lsb, m, pawn, knight);//knight
				}else
					push_move (lsb, m, pawn);
				//double move! for black it will be from rank 7.
				if (lsb & rank7)
				{
					m >>= 8;
					if (m & emptysq)
						push_move (lsb, m, pawn, 0, FLAGDOUBLEMOVE);
				}
			}
			//capture moves for black
			c = lsb >> 9;
			c &= ~fileh;
			if (c & epsq)
				push_move (lsb, c, 0, FLAGENPASSANT);
			c &= all[white];
			if (c)
			{
				if (c & rank1)//pawn is promoted to...
				{
					push_move (lsb, c, pawn, queen);//queen
					push_move (lsb, c, pawn, rook);//rook
					push_move (lsb, c, pawn, bishop);//bishop
					push_move (lsb, c, pawn, knight);//knight
				}else
					push_move (lsb, c, pawn);
			}
			c = lsb >> 7;
			c &= ~filea;
			if (c & epsq)
				push_move (lsb, c, pawn, 0, FLAGENPASSANT);
			c &= all[white];
			if (c)
			{
				if (c & rank1)//pawn is promoted to...
				{
					push_move (lsb, c, pawn, queen);//queen
					push_move (lsb, c, pawn, rook);//rook
					push_move (lsb, c, pawn, bishop);//bishop
					push_move (lsb, c, pawn, knight);//knight
				}else
					push_move (lsb, c, pawn);
			}
		}
		//remove the lsb from pawns bits
		ap ^= lsb;
	}
}
// generate king moves
void Engine::gen_king_moves(side movefor) {
    // squares not occupied by our pieces
	bitboard othersq = ~all[movefor];
	//get king position
	bitboard lsb = pieces[movefor][king];
	bitboard m;//moved to position
    int lsbint;
    bitboard ant/*all move to positions*/;
	lsbint = get_bit_pos(lsb);
	// get all moves on that position
	ant = king_moves[lsbint];

	// generate castling move if allowed
	// need to check for check
	if (!kingmoved[movefor]) {
	   // castle towards filea
	   if (!rookmoved[movefor][0]) {
          if (movefor==white && !(all[white]&0xE)) {
             push_move(lsb, 0x4, king, 0, FLAGCASTLEA, 0x1 | 0x8);
          } else if (movefor==black && !(all[black]&0xE00000000000000LLU)) {
             push_move(lsb, 0x400000000000000LLU, king, 0, FLAGCASTLEA, 0x100000000000000LLU | 0x800000000000000LLU);
          }
       }
       // castle towards fileh
       if (!rookmoved[movefor][1]) {
          if (movefor==white && !(all[white]&0x60)) {
             push_move(lsb, 0x40, king, 0, FLAGCASTLEH, 0x80 | 0x20);
          }else if (movefor==black && !(all[black]&0x6000000000000000LLU)) {
             push_move(lsb, 0x4000000000000000LLU, king, 0, FLAGCASTLEH, 0x8000000000000000LLU | 0x2000000000000000LLU);
          }
       }
    }
	// loop thru' all moves
	while (ant) {
          m = ant & (~ant + 1);
          if (m & othersq) {
             push_move (lsb, m, king);
          }
          ant ^= m;
    }
}
// generate knight moves
void Engine::gen_knight_moves(side movefor) {
    // squares not occupied by our pieces
	bitboard othersq = ~all[movefor];
	//get position
	bitboard an = pieces[movefor][knight];
	bitboard lsb;//last significant bit
	bitboard m;//moved to position
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
              if (m & othersq) {
                 push_move (lsb, m, knight);
              }
              ant ^= m;
        }
		//remove the lsb from pawns bits
		an ^= lsb;
    }
}
// generate rook moves
void Engine::gen_rook_moves(byte piecefor, bitboard ar, side movefor) {
    // squares not occupied by our pieces
	bitboard othersq = ~all[movefor];
	// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

	bitboard lsb;//last significant bit
	bitboard m;//moved to position
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
		_um = _um << 8 | _um << 16 | _um << 24 | _um << 32 | _um << 40 | _um << 48;
		_um &= up_moves[lsbint];
		_um ^= up_moves[lsbint];
		_um &= othersq;

        // generate moves to the bottom
		_dm = down_moves[lsbint] & _all;
		_dm = _dm >> 8 | _dm >> 16 | _dm >> 24 | _dm >> 32 | _dm >> 40 | _dm >> 48;
		_dm &= down_moves[lsbint];
		_dm ^= down_moves[lsbint];
		_dm &= othersq;

		// loop thru' all moves
		_dm = _dm | _um | _lm | _rm;
		while (_dm) {
              m = _dm & (~_dm + 1);
              if (m & othersq) {
                 push_move (lsb, m, piecefor);
              }
              _dm ^= m;
        }
		ar ^= lsb;
    }
}
// generate bishop moves
void Engine::gen_bishop_moves(byte piecefor, bitboard ab, side movefor){
    // squares not occupied by our pieces
	bitboard othersq = ~all[movefor];
	// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

	bitboard lsb;//last significant bit
	bitboard m;//moved to position
    int lsbint;
    bitboard _45m, _225m, _135m, _315m;
    //iterate thru' all bishops
	while (ab) {
		lsb = ab & (~ab + 1);
		lsbint = get_bit_pos(lsb);
		//generate moves for diagonally right up
		_45m = deg45_moves[lsbint] & _all;
		_45m = _45m << 9 | _45m << 18 | _45m << 27 | _45m << 36 | _45m << 45 | _45m << 54;
		_45m &= deg45_moves[lsbint];
		_45m ^= deg45_moves[lsbint];
		_45m &= othersq;

        // generate moves for left down
		_225m = deg225_moves[lsbint] & _all;
		_225m = _225m >> 9 | _225m >> 18 | _225m >> 27 | _225m >> 36 | _225m >> 45 | _225m >> 54;
		_225m &= deg225_moves[lsbint];
		_225m ^= deg225_moves[lsbint];
		_225m &= othersq;

        // generate moves right down
		_135m = deg135_moves[lsbint] & _all;
		_135m = _135m >> 7 | _135m >> 14 | _135m >> 21 | _135m >> 28 | _135m >> 35 | _135m >> 42;
		_135m &= deg135_moves[lsbint];
		_135m ^= deg135_moves[lsbint];
		_135m &= othersq;

        // generate moves for left up
		_315m = deg315_moves[lsbint] & _all;
		_315m = _315m << 7 | _315m << 14 | _315m << 21 | _315m << 28 | _315m << 35 | _315m << 42;
		_315m &= deg315_moves[lsbint];
		_315m ^= deg315_moves[lsbint];
		_315m &= othersq;

		// loop thru' all moves
		_315m = _315m | _135m | _225m | _45m;
		while (_315m) {
              m = _315m & (~_315m + 1);
              if (m & othersq) {
                 push_move (lsb, m, piecefor);
              }
              _315m ^= m;
        }
		ab ^= lsb;
    }
}
//generate moves for the given on the current board position
void Engine::generate_moves(side movefor) {
	//empty the stack
	//we have to put in fresh moves.
	stack.init();
	gen_king_moves (movefor);
	// rook and bishop moves for the queen
	gen_rook_moves (queen, pieces[movefor][queen], movefor);
	gen_bishop_moves (queen, pieces[movefor][queen], movefor);
    // actual rook and bishop moves
	gen_rook_moves (rook, pieces[movefor][rook], movefor);
	gen_bishop_moves (bishop, pieces[movefor][bishop], movefor);
	gen_knight_moves (movefor);
	gen_pawn_moves (movefor);
}

void Engine::list_moves() {
    for (int i=0; i<stack.top; i++)
        cout << stack.getMoveTxt(i) << ", ";
}
