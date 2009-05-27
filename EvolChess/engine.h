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
#include "debug.h"

//move structure to store moves
class cmove {
private:
    char mov[8];
public:
	bitboard from;//square from where piece is move
	bitboard to;//square where piece ends up
	bitboard mov2;//rook moves when castling
	byte piece;//piece that moved
	byte promotedto;//piece to which the pawn is promoted to
	byte flags;//tells us if the move is double or en passant
	byte captured;//which piece was captured

	cmove (cmove *m) { from=m->from; to=m->to;  mov2=m->mov2; piece=m->piece; promotedto=m->promotedto; flags=m->flags; captured=m->captured; }
	cmove (bitboard f, bitboard t, bitboard m2=0, byte p=0, byte pt=0, byte flg=0, byte c=0) {
		from = f; to = t; mov2 = m2; piece = p; promotedto = pt; flags = flg; captured = c;
	}
	char *getMoveTxt();
	int isequal(cmove *m) {
		if (from & m->from && to & m->to && piece==m->piece)
			return 1;
		return 0;
	}
};
//moves tree for ai
class MoveNode {
protected:
	void init() { move=NULL;child=NULL;next=NULL;score=0; }
public:
	// the move
	cmove *move;
	// moves that can be played after this one
	MoveNode *child;
	// siblings
	MoveNode *next;
	// static position score after this move
	int score;

	virtual ~MoveNode () { if (move) delete move; }

	MoveNode(){init();}
	MoveNode(cmove *m) { init(); move=m; }
	void addChild(cmove *m){if (!m) return;MoveNode *n = new MoveNode(m); n->next = child; child = n;}
};
class moveshist {
public:
	cmove *move;
	moveshist *prev;
	bitboard allpos;

	moveshist() { move=NULL; allpos = 0; }
	~moveshist() { if (move) delete move; }

	moveshist(cmove *m, bitboard b) { move = m; allpos = b; }
};
//dynamic stack for moves history
class dmovestack {
protected:
	moveshist *_top;
	int _size;
public:
	dmovestack () { init(); }
	void init() { _top = NULL; _size=0; while (_top!=NULL){ moveshist *m=_top; _top=_top->prev; delete m;} }
	void  push(moveshist *move) { move->prev=_top; _top=move; _size++;}
	moveshist *pop () { if (!_size) return NULL; _size--; moveshist *m=_top; _top=m->prev; return m; }
	int size() { return _size; }
	moveshist *lastmove() { return _top; }
};

class Engine {
private:
    cdebug debug;
    int prosnodes;
protected:
	// max depth for normal search
	static const int MAX_AI_SEARCH_DEPTH = 5;
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
	//mate flag
	int ismate;
	//3fold draw flag
	int isthreefoldw;
	int isthreefoldb;

	//this is the square where the adversary will capture
	//if move is e2e4 then epsq will be e3
	bitboard epsq;
	int kingmoved[2], rookmoved[2][2];

	//history of moves done
	dmovestack moveshistory;

	//the stack
	//cmovestack stack;

	//check if the user/xboard move is valid or not
	cmove *check_move (bitboard f, bitboard t, int promto);
	cmove *check_piece_move (bitboard f, bitboard t, int promto, int cap);
	bitboard gen_rook_moves2(bitboard rp, side movefor);
	bitboard gen_bishop_moves2(bitboard rp, side movefor);

	//generate next moves
	cmove *create_move(bitboard f, bitboard t, byte moved, byte promto, byte flags, bitboard mov2);
	void generate_moves(MoveNode *par);
	void gen_pawn_moves(MoveNode *par);
	void gen_king_moves(MoveNode *par);
	void gen_knight_moves(MoveNode *par);
	void gen_rook_moves(byte piecefor, bitboard ar, MoveNode *par);
	void gen_bishop_moves(byte piecefor, bitboard ar, MoveNode *par);

	void gen_atk_moves(side moveof, bitboard& atkbrd);
	void gen_king_atk(side movefor, bitboard& atkbrd);
	void gen_pawn_atk(side moveof, bitboard& atkbrd);
	void gen_knight_atk(side moveof, bitboard& atkbrd);
	void gen_rook_atk (side moveof, bitboard& atkbrd);
	void gen_bishop_atk (side moveof, bitboard& atkbrd);

	int next_ply_best_score(MoveNode *par, int depth, int alpha, int beta, MoveNode *bm);
	MoveNode *insert_sort(MoveNode *par, MoveNode *c);
	int static_position_score();
	void cleannode(MoveNode *node);

	int checkfordraw();
public:
	int gameended;

	Engine();
	virtual ~Engine();

	void newGame();
	int isMateMove() { return ismate; }
	int isDraw() { if (isthreefoldw==2 || isthreefoldb==2) return 1; return 0; }
	void show_board();
	void list_moves();

	cmove *doaimove();

	int sidetomove() { return moveof; };
	void domove (cmove *move);
	void undomove(cmove *move);
	void undolastmove();
	cmove *input_move(char *m);

	int get_bit_pos(bitboard b);
};


#endif /* ENGINE_H_ */
