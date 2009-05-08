/* * * * * * * * * *
 * EvolChess
 * * * * * * * * * *
 * engine.h
 *
 *  Created on: May 1, 2009
 *      Author: Bhupendra Aole
 */

#ifndef ENGINE_H_
#define ENGINE_H_

#include "constants.h"

//move structure to store moves
class cmove
{
public:
	cmove *_prev;

	bitboard from;//square from where piece is move
	bitboard to;//square where piece ends up
	bitboard mov2;//rook moves when castling
	byte piece;//piece that moved
	byte promotedto;//piece to which the pawn is promoted to
	byte flags;//tells us if the move is double or en passant
	byte captured;//which piece was captured

	cmove(){}
	cmove(cmove *m) { from=m->from; to=m->to;  mov2=m->mov2; piece=m->piece; promotedto=m->promotedto; flags=m->flags; captured=m->captured; }
};

//dynamic stack for moves history
class dmovestack {
protected:
	cmove *_top;
	int _size;
public:
	dmovestack () { init(); }
	void init() { _top = NULL; _size=0; while (_top!=NULL){ cmove *m=_top; _top=_top->_prev; delete m;} }
	void  push(cmove move) { cmove *m = new cmove(move); m->_prev=_top; _top=m; _size++;}
	cmove pop () { if (!_size) return NULL; _size--; cmove *m=_top; _top=m->_prev; return m; }
	int size() { return _size; }
	cmove lastmove() { return *_top; }
};

/*stack of moves
We dont really need it now but it will be
helpfull in future version where we will
implement computer play (AI)
*/
class cmovestack
{
public:
	int top;//to track how many moves are already in

	cmove move[100];//max no. of moves possible at any position: we'll have to inc. it later.
	int positionvalue[100];

	bitboard allMoves;//all moves of all the pieces.
    char mov[8];

    void init() {
         allMoves = 0;
         top = 0;
    }
	//store a move in stack and increment the top
	void push(bitboard f, bitboard t, byte moved, byte captured, byte promto, byte flags, bitboard mov2);
	void push(cmove m) { move[top++] = m; };

	//pop the last move
	cmove pop(){ return move[--top]; };
	cmove lastmove() { return move[top-1]; };

	//find if the move exists in the stack
	int find (bitboard f, bitboard t, int promto);

	char *getMoveTxt(int index);
};

class Engine {
protected:
	// board to store position of all black n white pieces
	bitboard all[2];
	// board to store position of each type of piece
	bitboard pieces[2][6];

	//side to move
	side moveof;
	//side not to move
	side movenotof;
	//move number
	byte moveno;

	//this is the square where the adversary will capture
	//if move is e2e4 then epsq will be e3
	bitboard epsq;
	int kingmoved[2], rookmoved[2][2];

	//history of moves done
	dmovestack moveshistory;

	//the stack
	cmovestack stack;

	void push_move(bitboard f, bitboard t, byte moved, byte promto, byte flags, bitboard mov2);
	void generate_moves(side moveof);
	void gen_pawn_moves(side movefor);
	void gen_king_moves(side movefor);
	void gen_knight_moves(side movefor);
	void gen_rook_moves(byte piecefor, bitboard ar, side movefor);
	void gen_bishop_moves(byte piecefor, bitboard ar, side movefor);

	void gen_atk_moves(side moveof, bitboard& atkbrd);
	void gen_king_atk(side movefor, bitboard& atkbrd);
	void gen_pawn_atk(side moveof, bitboard& atkbrd);
	void gen_knight_atk(side moveof, bitboard& atkbrd);
	void gen_rook_atk (side moveof, bitboard& atkbrd);
	void gen_bishop_atk (side moveof, bitboard& atkbrd);
public:
	int gameended;

	Engine();
	virtual ~Engine();

	void newGame();
	void show_board();

	int sidetomove() { return moveof; };
	int domove(int index);
	void domove (cmove move);
	int doaimove();
	void undomove(cmove m);
	void undolastmove();
	int input_move(char *m);

	int get_bit_pos(bitboard b);
	void generate_moves() { generate_moves(moveof); };

	void list_moves();
	char *getMoveTxt(int index) { return stack.getMoveTxt(index); }
};


#endif /* ENGINE_H_ */
