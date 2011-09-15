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

//move structure to store book moves
class simplemove {
public:
	bitboard move;
	simplemove *next;
	simplemove *sibling;
	int sibcnt;

	simplemove(bitboard m) {
		move = m;
		next = NULL;
		sibling = NULL;
		sibcnt = 0;
	}
};

//move structure to store moves
class cmove {
private:
	char mov[8];
public:
	static int cnt;
	static cmove *top;

	bitboard from; //square from where piece is move
	bitboard to; //square where piece ends up
	bitboard mov2; //rook moves when castling
	byte piece; //piece that moved
	byte promotedto; //piece to which the pawn is promoted to
	byte flags; //tells us if the move is double or en passant
	byte captured; //which piece was captured

	int score;

	cmove *next;

	cmove() {
		cmove(0, 0, 0, 0, 0, 0);
	}
	cmove(cmove *m) {
		cmove(m->from, m->to, m->mov2, m->piece, m->promotedto, m->flags,
				m->captured);
		//cout<<"m:"<<getMoveTxt()<<endl;
	}
	cmove(bitboard f, bitboard t, bitboard m2 = 0, byte p = 0, byte pt = 0,
			byte flg = 0, byte c = 0) {
		cnt++;
		from = f;
		to = t;
		mov2 = m2;
		piece = p;
		promotedto = pt;
		flags = flg;
		captured = c;
		next = NULL;
		//cout<<"m:"<<getMoveTxt()<<endl;
	}
	~cmove() {
		cnt--;
		//cout<<"d:"<<getMoveTxt()<<endl;
	}

	void set(bitboard f, bitboard t, bitboard m2, byte p, byte pt, byte flg,
			byte c) {
		from = f;
		to = t;
		mov2 = m2;
		piece = p;
		promotedto = pt;
		flags = flg;
		captured = c;
		next = NULL;
	}

	void copy(cmove &m) {
		from = m.from;
		to = m.to;
		mov2 = m.mov2;
		piece = m.piece;
		promotedto = m.promotedto;
		flags = m.flags;
		captured = m.captured;
		next = NULL;
	}

	char *getMoveTxt();
	int isequal(cmove *m) {
		if ((from & m->from) && (to & m->to) && piece == m->piece)
			return 1;
		return 0;
	}

	static cmove *newcmove(bitboard f, bitboard t, bitboard m2 = 0, byte p = 0,
			byte pt = 0, byte flg = 0, byte c = 0) {
		if (!top) {
			return new cmove(f, t, m2, p, pt, flg, c);
		}
		cmove *m = top;
		top = top->next;
		m->set(f, t, m2, p, pt, flg, c);

		return m;
	}

	static cmove *newcmove(cmove *m) {
		return newcmove(m->from, m->to, m->mov2, m->piece, m->promotedto,
				m->flags, m->captured);
	}

	static void deletecmove(cmove *m) {
		if (!m)
			return;
		m->next = top;
		top = m;
	}
	static void gc() {
		cmove *m;
		while (top) {
			m = top;
			top = top->next;
			delete m;
		}
	}
};

//moves tree for ai
class MoveNode {
private:
	int id;
protected:
	void init() {
		move = NULL;
		child = NULL;
		next = NULL;
		score = 0;
	}
public:
	// the move
	cmove *move;
	// moves that can be played after this one
	MoveNode *child;
	// siblings
	MoveNode *next;
	// static position score after this move
	int score;

	virtual ~MoveNode() {
		//cout<<id<<",";
		if (child)
			delete child;
		if (next)
			delete next;

		cmove::deletecmove(move);
	}

	MoveNode() {
		init();
	}

	MoveNode(cmove *m) {
		init();
		move = m;
	}
	void addChild(cmove *m) {
		if (!m)
			return;
		MoveNode *n = new MoveNode(m);
		n->score = m->score;
		n->next = child;
		child = n;
	}
};

class moveshist {
private:
	int id;
public:
	cmove *move;
	moveshist *prev;
	bitboard allpos;
	static int cnt;

	moveshist() {
		moveshist(NULL, 0);
	}
	~moveshist() {
		//delete move;
		//cout<<"D:"<<id<<",";
	}

	moveshist(cmove *m, bitboard b) {
		move = m; //new cmove(m);
		allpos = b;
		id = cnt++;
		//cout<<"N:"<<id<<",";
	}
};

//dynamic stack for moves history
class dmovestack {
protected:
	moveshist *_top;
	int _size;

public:
	dmovestack() {
		_top = NULL;
		_size = 0;
	}
	void init() {
		_size = 0;
		while (_top) {
			moveshist *m = _top;
			_top = _top->prev;
			delete m;
		}
	}
	void push(moveshist *move) {
		move->prev = _top;
		_top = move;
		_size++;
	}
	moveshist *pop() {
		if (!_size)
			return NULL;
		_size--;
		moveshist *m = _top;
		_top = m->prev;
		return m;
	}
	int size() {
		return _size;
	}
	moveshist *lastmove() {
		return _top;
	}
};

class PVLine2 {
public:
	int num; // Number of moves in the line.
	cmove argmove[10]; // The line.

	PVLine2() {
		num = 0;
	}
	void print() {
		for (int i = 0; i < num; i++) {
			cout << argmove[i].getMoveTxt() << " ";
		}
	}
};

class Engine {
private:
	cdebug debug;
	simplemove *bktop;
	simplemove *bkcurrent;
	time_t t1;
	int tnodes;
	int movescore;
	int movesintime, mps;
	int timeleft;

#ifdef DEBUG
	int stopsearch;
#endif
	//int trilen[8];
	//cmove triarr[8][8];

protected:
	// max depth for normal search
	static const int MAX_AI_SEARCH_DEPTH = 6;
	// board to store position of all black n white pieces
	bitboard all[2];
	// board to store position of each type of piece
	bitboard pieces[2][6];

	//side to move
	side moveof;
	//side not to move
	side movenotof;
	//move number
	int moveno;
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
	//current best move
	//cmove *bestmove;

	//the stack
	//cmovestack stack;

	//check if the user/xboard move is valid or not
	cmove *check_move(bitboard f, bitboard t, int promto);
	cmove *check_piece_move(bitboard f, bitboard t, int promto, int cap);
	bitboard gen_rook_moves2(bitboard rp, side movefor);
	bitboard gen_bishop_moves2(bitboard rp, side movefor);

	//generate next moves
	cmove *create_move(bitboard mov);
	cmove *create_move(bitboard f, bitboard t, byte moved, byte promto,
			byte flags, bitboard mov2);
	void generate_moves(MoveNode *par);
	void gen_pawn_moves(MoveNode *par);
	void gen_king_moves(MoveNode *par);
	void gen_knight_moves(MoveNode *par);
	void gen_rook_moves(byte piecefor, bitboard ar, MoveNode *par);
	void gen_bishop_moves(byte piecefor, bitboard ar, MoveNode *par);
/*
	void gen_atk_moves(side moveof, bitboard& atkbrd);
	void gen_king_atk(side movefor, bitboard& atkbrd);
	void gen_pawn_atk(side moveof, bitboard& atkbrd);
	void gen_knight_atk(side moveof, bitboard& atkbrd);
	void gen_rook_atk(side moveof, bitboard& atkbrd);
	void gen_bishop_atk(side moveof, bitboard& atkbrd);
*/

	void gen_atk_moves( bitboard& atkbrd);
	void gen_king_atk(bitboard& atkbrd);
	void gen_pawn_atk(bitboard& atkbrd);
	void gen_knight_atk( bitboard& atkbrd);
	void gen_rook_atk(bitboard& atkbrd);
	void gen_bishop_atk(bitboard& atkbrd);

	int alphabeta(int ply, int depth, int alpha, int beta,
			PVLine2 *pline);
	int qs(int depth, int alpha, int beta);

	MoveNode *insert_sort(MoveNode *par, MoveNode *c);
	int evaluate();

	int checkfordraw();

public:
	int gameended;

	Engine();
	virtual ~Engine();

	void init();
	void loadDefaultBook();
	void newGame();
	int isMateMove() {
		return ismate;
	}
	int isDraw() {
		if (isthreefoldw == 2 || isthreefoldb == 2)
			return 1;
		return 0;
	}
	void show_board();
	void list_moves();
	bitboard getBitMove(string m);

	void aimove(cmove &move);

	int sidetomove() {
		return moveof;
	}
	;
	void domove(cmove *move);
	void undomove(cmove *move);
	void undolastmove();
	int input_move(char *m);

	int get_bit_pos(bitboard b);

	void inittime(char *t);
	void setowntime(char *t);
};

#endif /* ENGINE_H_ */
