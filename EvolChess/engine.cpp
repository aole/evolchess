/* * * * * * * * * *
 * EvolChess
 * * * * * * * * * *
 * engine.cpp
 *
 *  Created on: May 1, 2009
 *      Author: Bhupendra Aole
 */

#include <iostream>
#include <time.h>
#include <sstream>
#include <algorithm>
#include <iomanip>

#include "engine.h"
#include "debug.h"
#include "windows.h"

using namespace std;

cmove *cmove::top = NULL;
int moveshist::cnt = 0;

char *cmove::getMoveTxt() {
	if (from & filea)
		mov[0] = 'a';
	else if (from & fileb)
		mov[0] = 'b';
	else if (from & filec)
		mov[0] = 'c';
	else if (from & filed)
		mov[0] = 'd';
	else if (from & filee)
		mov[0] = 'e';
	else if (from & filef)
		mov[0] = 'f';
	else if (from & fileg)
		mov[0] = 'g';
	else if (from & fileh)
		mov[0] = 'h';

	if (from & rank1)
		mov[1] = '1';
	else if (from & rank2)
		mov[1] = '2';
	else if (from & rank3)
		mov[1] = '3';
	else if (from & rank4)
		mov[1] = '4';
	else if (from & rank5)
		mov[1] = '5';
	else if (from & rank6)
		mov[1] = '6';
	else if (from & rank7)
		mov[1] = '7';
	else if (from & rank8)
		mov[1] = '8';

	if (to & filea)
		mov[2] = 'a';
	else if (to & fileb)
		mov[2] = 'b';
	else if (to & filec)
		mov[2] = 'c';
	else if (to & filed)
		mov[2] = 'd';
	else if (to & filee)
		mov[2] = 'e';
	else if (to & filef)
		mov[2] = 'f';
	else if (to & fileg)
		mov[2] = 'g';
	else if (to & fileh)
		mov[2] = 'h';

	if (to & rank1)
		mov[3] = '1';
	else if (to & rank2)
		mov[3] = '2';
	else if (to & rank3)
		mov[3] = '3';
	else if (to & rank4)
		mov[3] = '4';
	else if (to & rank5)
		mov[3] = '5';
	else if (to & rank6)
		mov[3] = '6';
	else if (to & rank7)
		mov[3] = '7';
	else if (to & rank8)
		mov[3] = '8';

	if (promotedto < 5 && promotedto > 0) {
		mov[4] = notationb[promotedto][0];
		mov[5] = '\0';
	} else
		mov[4] = '\0';

	return mov;
}

Engine::Engine() {
	debug.open_debug_file();
}

Engine::~Engine() {
	debug.close_debug_file();
}

// initialize default book
void Engine::loadDefaultBook() {
	string line;
	//check if file default.ectb is present
	//open the file
	ifstream fbk;
	fbk.open("default.ectb");
	if (!fbk.is_open()) {
		bktop = bkcurrent = __null;
		return;
	}
	//set top and current pointers
	bktop = bkcurrent = new simplemove(0);
	simplemove *_simm;
	string _sm;
	bitboard _mv;
	while (fbk.good()) {
		getline(fbk, line);
		istringstream _ss(line);
		bkcurrent = bktop;
		while (getline(_ss, _sm, ' ')) {
			_mv = getBitMove(_sm);
			_simm = bkcurrent->next;
			while (_simm != __null) {
				if (_mv == _simm->move)
					break;
				_simm = _simm->sibling;
			}
			if (_simm == __null) {
				_simm = new simplemove(_mv);
				if (bkcurrent->next == __null) {
					bkcurrent->next = _simm;
					_simm->sibcnt++;
				} else {
					_simm->sibling = bkcurrent->next->sibling;
					bkcurrent->next->sibling = _simm;
					bkcurrent->next->sibcnt++;
				}
			}
			bkcurrent = _simm;
		}
	}
	//read each line
	fbk.close();
}
//initializes the engine
void Engine::init() {
	loadDefaultBook();
}

//initializes the game to starting position
void Engine::newGame() {
	ismate = 0;
	isthreefoldw = isthreefoldb = 0;
	//reset book
	bkcurrent = bktop;
	//stack.init();
	moveshistory.init();
	moveno = 0;
	gameended = 0;
	isendgame = 0;
	//place all white and black pieces
	for (int s = 0; s < 2; s++) {
		all[s] = start_all[s];
		for (int p = 0; p < 6; p++)
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
void Engine::show_board() {
	cout << "\n";
	bitboard p = 0;
	char toprint[2];

	for (int r = 8; r > 0; r--) {
		cout << r << " ";
		for (int f = 0; f < 8; f++) {
			p = rank[r - 1] & file[f];
			strcpy(toprint, "*");
			for (int i = 0; i < 6; i++)
				if (pieces[white][i] & p) {
					strcpy(toprint, notationw[i]);
					break;
				} else if (pieces[black][i] & p) {
					strcpy(toprint, notationb[i]);
					break;
				}
			cout << toprint << " ";
		}
		cout << endl;
	}
	cout << "  ";
	for (char f = 'a'; f <= 'h'; f++)
		cout << f << " ";
	cout << endl;
}

void Engine::undolastmove() {
	undomove(moveshistory.lastmove()->move);
}
//undo last move
/*1. switch sides
 2. pop move from history stack
 4. if rook was moved first time unmark moveno
 5. if king was moved first time unmark moveno
 6. if enpassent add all and pawn
 7. add individual piece
 8. add all
 9. undo second move individual piece
 10. undo second move all
 11. undo individual move
 12. undo all
 */
void Engine::undomove(cmove *move) {
	//cout<<"undo "<<move->getMoveTxt()<<endl;
	cmove *lm = NULL;
	bitboard t = move->to, f = move->from;
	bitboard mov = f | t;
	//switch move of
	moveof = movenotof;
	movenotof = (moveof == white ? black : white);

	//if (!istemp)
	delete moveshistory.pop();

	if (moveof == white)
		if (isthreefoldw)
			isthreefoldw--;
	if (moveof == black)
		if (isthreefoldb)
			isthreefoldb--;

	//for castling purpose
	if (move->piece == king && kingmoved[moveof] == moveno)
		kingmoved[moveof] = 0;
	if (move->piece == rook && (f & filea) && rookmoved[moveof][0] == moveno)
		rookmoved[moveof][0] = 0;
	else if (move->piece == rook && (f & fileh)
			&& rookmoved[moveof][1] == moveno)
		rookmoved[moveof][1] = 0;

	//enpassent
	if (move->flags & FLAGENPASSANT) {
		if (moveof == white) {
			all[movenotof] ^= t >> 8; //the pawn to be captured is actually, one rank back
			pieces[movenotof][pawn] ^= t >> 8;
		} else {
			all[movenotof] ^= t << 8; //the pawn to be captured is actually, one rank back
			pieces[movenotof][pawn] ^= t << 8;
		}
	} else if (move->captured) {
		//respawn captured piece
		all[movenotof] ^= t;
		pieces[movenotof][move->captured] ^= t;
	}
	//set enpassant square denoted by last move
	epsq = 0;
	lm = moveshistory.lastmove() ? moveshistory.lastmove()->move : NULL;
	if (lm) {
		if (lm->flags & FLAGDOUBLEMOVE) {
			if (moveof == white) //last move was black
				epsq = lm->to << 8;
			else
				epsq = lm->to >> 8;
		}
	}
	//undo move
	all[moveof] ^= mov;
	if (move->promotedto) {
		pieces[moveof][move->piece] ^= f;
		pieces[moveof][move->promotedto] ^= t;
	} else
		pieces[moveof][move->piece] ^= mov;

	//undo II move if there
	if (move->mov2) {
		all[moveof] ^= move->mov2;
		pieces[moveof][rook] ^= move->mov2;
	}
	moveno--;
}

// ai with Negamax Search
void Engine::aimove(cmove &move) {
	cmove *bkmove;
	simplemove *_sm;

//	list_moves();

	// check if book move is available
	if (bkcurrent != NULL) {
		_sm = bkcurrent->next;
		if (_sm != NULL) {
			int rnd = rand() % _sm->sibcnt;
			for (int i = 0; i < _sm->sibcnt; i++) {
				if (i == rnd)
					break;
				_sm = _sm->sibling;
			}
			bkcurrent = _sm;
			bkmove = create_move(_sm->move);
			move.copy(*bkmove);
			return;
		}
	}
	//HANDLE th = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) findbestmove,
	//		NULL, 0, NULL);
	//WaitForSingleObject(th, INFINITE);
	//CloseHandle(th);

	PVLine2 line;
	int score;

	// time for this move
	// if prev. search comleted in half the time left for the move
	// do another search with inc. depth.
	if (moveno >= movesintime)
		movesintime += mps;
	int movestodo = movesintime - moveno;
	int htmlft = timeleft / movestodo;
	cout << "# movesintime:" << movesintime << ", movestodo:" << movestodo
			<< ", half time left:" << htmlft << endl;

	//start timer
	t1 = clock();
	int depth = 2; //MAX_AI_SEARCH_DEPTH;
	//int window = 50,
	int alpha = -VALUEINFINITE, beta = VALUEINFINITE;

	tnodes = 0;
	stopsearch = 0;
	while (1) {
		int ply = 0;
		score = alphabeta(ply, depth, alpha, beta, &line);
		// mating move found
		// on need to search further
		if (score >= piecevalue[king] || stopsearch)
			break;

		time_t t2 = clock() - t1;
		if (t2 < 1500)
			t2 *= 8;
		if (t2 > htmlft)
			break;
		depth++;
		/*// aspiration window
		 if ((score <= alpha)||(score >= beta)) {
		 alpha = -VALUEINFINITE;
		 beta = VALUEINFINITE;
		 continue;
		 }
		 alpha = score - window;
		 beta = score + window;
		 depth++;*/
	}
	move.copy(line.argmove[0]);
}

MoveNode *Engine::insert_sort(MoveNode *par, MoveNode *c) {
	MoveNode *prev = NULL;
	MoveNode *toreturn = c->next;

	c->next = NULL;

	if (!par->child) {
		par->child = c;
		return toreturn;
	}
	for (MoveNode *cur = par->child; cur; cur = cur->next) {
		if (c->weight > cur->weight) {
			c->next = cur;
			if (prev)
				prev->next = c;
			else
				par->child = c;
			return toreturn;
		}
		prev = cur;
	}
	prev->next = c;
	return toreturn;
}

int Engine::checkfordraw() {
	// check for 3fold move
	moveshist *cur = moveshistory.lastmove();
	bitboard p1, p2;
	if (moveshistory.size() > 8) {
		int i = 4;
		while (i--)
			cur = cur->prev;
		p1 = cur->allpos;
		i = 4;
		while (i--)
			cur = cur->prev;
		p2 = cur->allpos;

		if (p1 == p2 && p1 == moveshistory.lastmove()->allpos)
			return 1;
	}
	return 0;
}

void weigh_moves(vector<cmove*> &v, PVLine2 *line, int ply) {
	for (unsigned int i = 0; i < v.size(); i++) {
		if (line->num > ply) {
			if (line->argmove[ply].isequal(v[i])) {
				v[i]->weight += 1000;
			}
		}
	}
}

inline bool order_moves(cmove *m1, cmove *m2) {
	return m1->weight < m2->weight;
}

// is the position attached from "from" side.
// my side is self

int Engine::isUnderAttack(bitboard pos, side from, side self) {
	int ipos = get_bit_pos(pos);

	// any knight is attacking the position?
	if (knight_atk_brd(ipos, self) & (pieces[from][knight]))
		return 1;

	// opp. king is attacking my position?
	if (king_atk_brd(ipos, self) & (pieces[from][king]))
		return 1;

	// any pawn is attacking the position?
	if (pawn_atk_brd(pos, self) & (pieces[from][pawn]))
		return 1;

	// any rook/queen is attacking the position?
	if (rook_atk_brd(ipos, self) & (pieces[from][rook] | pieces[from][queen]))
		return 1;

	// any bishop/queen is attacking the position?
	if (bishop_atk_brd(ipos, self)
			& (pieces[from][bishop] | pieces[from][queen]))
		return 1;

	return 0;
}

int Engine::isBoardLegal(cmove *m) {
	bitboard atkmoves = 0;
	gen_atk_moves(atkmoves);
	if (pieces[movenotof][king] & atkmoves)
		return 0;
	//check castling
	//else if (cur->move->flags & FLAGCASTLEA) {
	else if (m->flags & FLAGCASTLEA) {
		if (moveof == black && (0x18 & atkmoves))
			return 0;
		else if (moveof == white && (0x1800000000000000ULL & atkmoves))
			return 0;
		//} else if (cur->move->flags & FLAGCASTLEH) {
	} else if (m->flags & FLAGCASTLEH) {
		if (moveof == black && (0x30 & atkmoves))
			return 0;
		else if (moveof == white && (0x3000000000000000ULL & atkmoves))
			return 0;
	}
	return 1;
}

// quiescent search
int Engine::qs(int alpha, int beta, PVLine2 *pline) {
	PVLine2 line;

	int cur_score = evaluate();

	if (cur_score >= beta)
		return beta;

	if (cur_score > alpha)
		return cur_score;

	vector<cmove*> v;
	generate_captures(v);

	if (v.empty()) {
		//pline->num = 0;
		return cur_score;
	}

	int fFoundPv = 0;
	int betafound = 0;
	cmove *m;

	while (!v.empty()) {
		m = v.back();
		if (betafound) {
			v.pop_back();
			cmove::deletecmove(m);
			continue;
		}
		domove(m);
		// check if move is legal
		if (isUnderAttack(pieces[movenotof][king], moveof, movenotof)) {
			//if (!isBoardLegal(m)) {
			undomove(m);
			v.pop_back();
			cmove::deletecmove(m);
			continue;
		}
		// move is legal
		tnodes++;

		if (fFoundPv) {
			cur_score = -qs(-beta, -alpha, &line);
			if ((cur_score > alpha) && (cur_score < beta))
				cur_score = -qs(-beta, -alpha, &line);
		} else
			cur_score = -qs(-beta, -alpha, &line);
		undomove(m);

		if (cur_score >= beta) {
			alpha = beta;
			betafound = 1;
		}
		if (cur_score > alpha) {
			alpha = cur_score;
			fFoundPv = 1;

			pline->argmove[0].copy(*m);
			memcpy(pline->argmove + 1, line.argmove, line.num * sizeof(cmove));
			pline->num = line.num + 1;
		}
		v.pop_back();
		cmove::deletecmove(m);
	}

	return alpha;
}

// alphabeta
int Engine::alphabeta(int ply, int depth, int alpha, int beta, PVLine2 *pline) {
	PVLine2 line;
	int cur_score;

	/*if (checkfordraw()){
	 pline->num = 0;
	 return 25; // contempt factor
	 }*/

	if (!depth) {
		pline->num = 0;
		return qs(alpha, beta, pline);
	}

	int childcount = 0;
	vector<cmove*> v;
	generate_pseudo_moves(v);
	weigh_moves(v, pline, ply);
	sort(v.begin(), v.end(), order_moves);

	int fFoundPv = 0;
	int betafound = 0;
	cmove *m;

	while (!v.empty()) {
		m = v.back();
		if (betafound) {
			v.pop_back();
			cmove::deletecmove(m);
			continue;
		}
		domove(m);
		// check if move is legal
		if (isUnderAttack(pieces[movenotof][king], moveof, movenotof)) {
			//if (!isBoardLegal(m)) {
			undomove(m);
			v.pop_back();
			cmove::deletecmove(m);
			continue;
		}
		// move is legal
		childcount++;

		if (fFoundPv) {
			cur_score = -alphabeta(ply + 1, depth - 1, -alpha - 1, -alpha,
					&line);
			if ((cur_score > alpha) && (cur_score < beta))
				cur_score = -alphabeta(ply + 1, depth - 1, -beta, -alpha,
						&line);
		} else
			cur_score = -alphabeta(ply + 1, depth - 1, -beta, -alpha, &line);
		undomove(m);

		if (cur_score >= beta) {
			alpha = beta;
			betafound = 1;
		}
		if (cur_score > alpha) {
			alpha = cur_score;
			pline->argmove[0].copy(*m);
			memcpy(pline->argmove + 1, line.argmove, line.num * sizeof(cmove));
			pline->num = line.num + 1;
			if (!ply) {
				time_t t = (clock() - t1) / 10;
				cout << depth << " " << cur_score << " " << t << " " << tnodes
						<< " ";
				pline->print();
				cout << endl;
			}
			fFoundPv = 1;
		}
		v.pop_back();
		cmove::deletecmove(m);
	}

	tnodes += childcount;
	if (!childcount) {
		return -piecevalue[king] * depth;
	} else if (childcount == 1 && ply == 0) {
		stopsearch = 1;
	}

	return alpha;
}

int Engine::evaluate() {
//calculate board value
	int bv[] = { 0, 0 }, otherside;
	bitboard lsb; //last significant bit

	for (int i = 0; i < 2; i++) {
		if (i == 0)
			otherside = 1;
		else
			otherside = 0;

		//piece value
		bitboard ap = all[i];
		while (ap) {
			lsb = ap & (~ap + 1);
			for (int j = 0; j < 6; j++) {
				if (lsb & pieces[i][j]) {
					bv[i] += piecevalue[j];
					break;
				}
			}
			ap ^= lsb;
		}

		// has end game started?
		if (!isendgame)
			isendgame = bv[i] < (1800 + piecevalue[king]);
	}

	// exchange pieces when ahead
	if (bv[white] > bv[black]) {
		bv[white] += (bv[white] / bv[black]) * 10;
	} else if (bv[black] > bv[white]) {
		bv[black] += (bv[black] / bv[white]) * 10;
	}

	for (int i = 0; i < 2; i++) {
		// focal center e4 d4 e5 d5
		// pawns
		if (pieces[i][pawn] & e4)
			bv[i] += 40;
		if (pieces[i][pawn] & d4)
			bv[i] += 40;
		if (pieces[i][pawn] & e5)
			bv[i] += 40;
		if (pieces[i][pawn] & d5)
			bv[i] += 40;
		// minor pieces
		if (pieces[i][knight] & e4)
			bv[i] += 20;
		if (pieces[i][knight] & d4)
			bv[i] += 20;
		if (pieces[i][knight] & e5)
			bv[i] += 20;
		if (pieces[i][knight] & d5)
			bv[i] += 20;
		if (pieces[i][bishop] & e4)
			bv[i] += 20;
		if (pieces[i][bishop] & d4)
			bv[i] += 20;
		if (pieces[i][bishop] & e5)
			bv[i] += 20;
		if (pieces[i][bishop] & d5)
			bv[i] += 20;
		// queen
		if (pieces[i][queen] & e4)
			bv[i] += 30;
		if (pieces[i][queen] & d4)
			bv[i] += 30;
		if (pieces[i][queen] & e5)
			bv[i] += 30;
		if (pieces[i][queen] & d5)
			bv[i] += 30;

		// pieces on wider center
		if (pieces[i][knight] & widercenter)
			bv[i] += 10;
		if (pieces[i][bishop] & widercenter)
			bv[i] += 10;
		if (pieces[i][rook] & widercenter)
			bv[i] += 10;
		if (pieces[i][queen] & widercenter)
			bv[i] += 10;

		// rook on central 4th n 5th rank
		if (pieces[i][rook] & (c4 | d4 | e4 | f4 | c5 | d4 | e5 | f5))
			bv[i] += 30;
		// rook on semi-open/open file
		if ((pieces[i][rook] & filea) && !(pieces[i][pawn] & filea)) {
			if (pieces[otherside][pawn] & filea)
				bv[i] += 20;
			else
				bv[i] += 30;
		}
		if ((pieces[i][rook] & fileb) && !(pieces[i][pawn] & fileb)) {
			if (pieces[otherside][pawn] & fileb)
				bv[i] += 20;
			else
				bv[i] += 30;
		}
		if ((pieces[i][rook] & filec) && !(pieces[i][pawn] & filec)) {
			if (pieces[otherside][pawn] & filec)
				bv[i] += 20;
			else
				bv[i] += 30;
		}
		if ((pieces[i][rook] & filed) && !(pieces[i][pawn] & filed)) {
			if (pieces[otherside][pawn] & filed)
				bv[i] += 20;
			else
				bv[i] += 30;
		}
		if ((pieces[i][rook] & filee) && !(pieces[i][pawn] & filee)) {
			if (pieces[otherside][pawn] & filee)
				bv[i] += 20;
			else
				bv[i] += 30;
		}
		if ((pieces[i][rook] & filef) && !(pieces[i][pawn] & filef)) {
			if (pieces[otherside][pawn] & filef)
				bv[i] += 20;
			else
				bv[i] += 30;
		}
		if ((pieces[i][rook] & fileg) && !(pieces[i][pawn] & fileg)) {
			if (pieces[otherside][pawn] & fileg)
				bv[i] += 20;
			else
				bv[i] += 30;
		}
		if ((pieces[i][rook] & fileh) && !(pieces[i][pawn] & fileh)) {
			if (pieces[otherside][pawn] & fileh)
				bv[i] += 20;
			else
				bv[i] += 30;
		}

		if (i == white) {
			// controlling focal center
			// pawns
			if (pieces[i][pawn] & c3)
				bv[i] += 10;
			if (pieces[i][pawn] & d3)
				bv[i] += 10;
			if (pieces[i][pawn] & e3)
				bv[i] += 10;
			if (pieces[i][pawn] & f3)
				bv[i] += 10;

			// blocked pawns
			if ((all[i] & d3) && (pieces[i][pawn] & d2))
				bv[i] -= 20;
			if ((all[i] & e3) && (pieces[i][pawn] & e2))
				bv[i] -= 20;

			// pawn on fifth rank
			if (pieces[i][pawn] & rank5)
				bv[i] += 10;
			// pawn on sixth rank
			if (pieces[i][pawn] & rank6)
				bv[i] += 30;

			// minor piece on fifth rank
			if (pieces[i][knight] & rank5)
				bv[i] += 25;
			if (pieces[i][bishop] & rank5)
				bv[i] += 25;
			// minor on sixth rank
			if (pieces[i][knight] & rank6)
				bv[i] += 50;
			if (pieces[i][bishop] & rank6)
				bv[i] += 50;

			// blocked rook due to non-castling
			if ((pieces[i][king] & (f1 | g1)) && (pieces[i][rook] & (h1 | g1)))
				bv[i] -= 80;

			if (!isendgame) {
				// king safety
				if (pieces[i][king] & (a1 | b1 | g1 | h1))
					bv[i] += 100;
				if (pieces[i][king] & (f1 | c1))
					bv[i] -= 50;
				if (pieces[i][king] & (e1 | d1))
					bv[i] -= 100;

				// king going for a stroll
				if (pieces[i][king] & rank2)
					bv[i] -= 50;
				else if (pieces[i][king] & rank3)
					bv[i] -= 100;
				else if (pieces[i][king] & rank4)
					bv[i] -= 300;
				else if (pieces[i][king] & enemycamp[i])
					bv[i] -= 500;

				// bishop shelter
				if ((pieces[i][king] & g1) && (pieces[i][bishop] & g2))
					bv[i] += 30;
				if ((pieces[i][king] & b1) && (pieces[i][bishop] & b2))
					bv[i] += 30;

				// Unreasonable retreats & non-development
				if (pieces[i][bishop] & (b1 | c1 | f1 | g1))
					bv[i] -= 40;
			}
		} else {
			// controlling focal center
			// pawns
			if (pieces[i][pawn] & c6)
				bv[i] += 10;
			if (pieces[i][pawn] & d6)
				bv[i] += 10;
			if (pieces[i][pawn] & e6)
				bv[i] += 10;
			if (pieces[i][pawn] & f6)
				bv[i] += 10;

			// blocked pawns
			if ((all[i] & d6) && (pieces[i][pawn] & d7))
				bv[i] -= 20;
			if ((all[i] & e6) && (pieces[i][pawn] & e7))
				bv[i] -= 20;

			// pawn on fourth rank
			if (pieces[i][pawn] & rank4)
				bv[i] += 10;
			// pawn on third rank
			if (pieces[i][pawn] & rank3)
				bv[i] += 30;

			// minor piece on fifth rank
			if (pieces[i][knight] & rank4)
				bv[i] += 25;
			if (pieces[i][bishop] & rank4)
				bv[i] += 25;
			// minor on sixth rank
			if (pieces[i][knight] & rank3)
				bv[i] += 50;
			if (pieces[i][bishop] & rank3)
				bv[i] += 50;

			// blocked rook due to non-castling
			if ((pieces[i][king] & (f8 | g8)) && (pieces[i][rook] & (h8 | g8)))
				bv[i] -= 80;

			if (!isendgame) {
				// king safety
				if (pieces[i][king] & (a8 | b8 | g8 | h8))
					bv[i] += 100;
				if (pieces[i][king] & (f8 | c8))
					bv[i] -= 50;
				if (pieces[i][king] & (e8 | d8))
					bv[i] -= 100;
				// bishop shelter
				if ((pieces[i][king] & g8) && (pieces[i][bishop] & g7))
					bv[i] += 30;
				if ((pieces[i][king] & b8) && (pieces[i][bishop] & b7))
					bv[i] += 30;

				// king going for a stroll
				if (pieces[i][king] & rank7)
					bv[i] -= 50;
				else if (pieces[i][king] & rank6)
					bv[i] -= 100;
				else if (pieces[i][king] & rank5)
					bv[i] -= 300;
				else if (pieces[i][king] & enemycamp[i])
					bv[i] -= 500;

				// Unreasonable retreats & development
				if (pieces[i][bishop] & (b8 | c8 | f8 | g8))
					bv[i] -= 40;
			}
		}
	}

	return bv[moveof] - bv[movenotof];
}
/* domove
 * 1. move all
 * 1.5. if promoted then remove pawn and add new piece
 2. move individual piece
 3. do second move all
 4. do second move individual piece
 5. remove all
 6. remove individual piece
 7. if enpassent remove all & pawn
 8. if king moved first time mark move no.
 9. if rook moved first time mark move no.
 10. if double move push enpassent square stack else push 0
 11. push move to history stack
 12. switch sides
 *
 */
void Engine::domove(cmove *move) {
	bitboard f = move->from;
	bitboard t = move->to;
	bitboard mov = f | t, mov2 = move->mov2;
	byte pt = move->promotedto;
	//byte p = move->piece;
	byte c = move->captured;
	side mof = moveof;
	//movescore = move->score;

	/*
	 #ifdef DEBUG
	 if ((all[moveof] & pieces[moveof][move->piece])
	 != pieces[moveof][move->piece])
	 cout << "domove trouble\n";
	 if (!f & all[moveof])
	 cout << "move not of moveof\n";
	 if (!f & pieces[moveof][move->piece])
	 cout << "move not of piece\n";
	 if (c && !(t & all[movenotof]))
	 cout << "captured not of movenoof\n";
	 if (c && !(t & pieces[movenotof][c]))
	 cout << "captured not of piece\n";
	 #endif
	 */

	moveno++;
//8. if king moved first time mark move no.
// check if king was moved. cannot castle then
	if (move->piece == king && !kingmoved[mof])
		kingmoved[mof] = moveno;
//9. if rook moved first time mark move no.
// check if rook was moved. in that case, cannot castle with that rook
	if (move->piece == rook && (f & filea) && !rookmoved[mof][0])
		rookmoved[mof][0] = moveno;
	else if (move->piece == rook && (f & fileh) && !rookmoved[mof][1])
		rookmoved[mof][1] = moveno;

//1. move all
	all[mof] ^= mov;
//1.5. if promoted then remove pawn and add new piece
//2. move individual piece
	if (pt) {
		pieces[mof][pawn] ^= f;
#ifdef DEBUG
		if (move->piece != pawn)
			cout << "error@domove";
#endif
		pieces[mof][pt] |= t;
	} else
		pieces[mof][move->piece] ^= mov;

//3. do second move all
//4. do second move individual piece
	if (mov2) {
		all[mof] ^= mov2;
		pieces[mof][rook] ^= mov2;
	}
//7. if enpassent remove all & pawn
//check which piece was that and move it
//if this is enpassent then we need to adjust
//for the capture of pawn at the right place
	if (move->flags & FLAGENPASSANT) {
		if (mof == white) {
			all[movenotof] ^= t >> 8;
			pieces[movenotof][pawn] ^= t >> 8;
		} else {
			all[movenotof] ^= t << 8;
			pieces[movenotof][pawn] ^= t << 8;
		}
	} else if (c) {
		//5. remove all
		all[movenotof] ^= t;
		//6. remove individual piece
		pieces[movenotof][c] ^= t;
	}
	if (move->flags & FLAGDOUBLEMOVE) {
		if (mof == white)
			epsq = t >> 8;
		else
			epsq = t << 8;
	} else
		epsq = 0;

	bitboard allpos = mof == white ? all[white] : all[black];
//record the move
//if (!istemp)
	moveshistory.push(new moveshist(move, allpos));

//change the sides
	moveof = movenotof;
	movenotof = mof; //(moveof == white ? black : white);
}

bitboard Engine::getBitMove(string m) {
	bitboard f = 1, t = 1;

	if (m[0] < 'a' || m[0] > 'h' || m[1] < '1' || m[1] > '8' || m[2] < 'a'
			|| m[2] > 'h' || m[3] < '1' || m[3] > '8')
		return 0;

//convert into int
	int from = ((m[1] - '1')) * 8;
	from += m[0] - 'a';
	int to = ((m[3] - '1')) * 8;
	to += m[2] - 'a';

//convert into bitboards
	f <<= from;
	t <<= to;

	return f | t;
}
//input that we got from the user
int Engine::input_move(char *m) {
//cout << "got move from user\n";
	bitboard f = 1, t = 1;
//initially we suppose its not a promotion move:
//cause its saved that way in cmove class
	byte pt = 0;

//check if string is legal
	/*m[0] and m[2] can be a, b, c, d, e, f, g, h
	 m[1] and m[3] can be 1, 2, 3, 4, 5, 6, 7, 8
	 */
	if (m[0] < 'a' || m[0] > 'h' || m[1] < '1' || m[1] > '8' || m[2] < 'a'
			|| m[2] > 'h' || m[3] < '1' || m[3] > '8')
		return 0;

//convert into int
	int from = ((m[1] - '1')) * 8;
	from += m[0] - 'a';
	int to = ((m[3] - '1')) * 8;
	to += m[2] - 'a';

//convert into bitboards
	f <<= from;
	t <<= to;

//if promotion then 'promoted to' required.
//m[4] can be q, r, b, n
	if (((f & pieces[white][pawn]) && moveof == white && (t & rank8))
			|| ((f & pieces[black][pawn]) && moveof == black && (t & rank1))) {
		if (strlen(m) < 5)
			return 0;
		if (m[4] == 'q')
			pt = queen;
		else if (m[4] == 'r')
			pt = rook;
		else if (m[4] == 'b')
			pt = bishop;
		else if (m[4] == 'n')
			pt = knight;
		else
			return 0;
	}
	cmove *mov = check_move(f, t, pt);

	if (mov)
		domove(mov);
	else
		return 0;
// check if book move
	if (bkcurrent != __null) {
		bkcurrent = bkcurrent->next;
		while (bkcurrent != __null) {
			if (bkcurrent->move & (f | t)) {
				break;
			}
			bkcurrent = bkcurrent->sibling;
		}
	}
	cmove::deletecmove(mov);
	return 1;
}

cmove *Engine::create_move(bitboard mov) {
	bitboard f = mov & all[moveof];
#ifdef DEBUG
	if (!f) {
		cout << "Illegal f:" << f;
	}
#endif
	bitboard t = mov ^ f;
#ifdef DEBUG
	if (!t) {
		cout << "Illegal t:" << t;
	}
#endif
	byte m = 0;

	for (int i = 0; i < 6; i++) {
		if (f & pieces[moveof][i]) {
			m = i;
			break;
		}
	}

	return create_move(f, t, m, 0, 0, 0);
}
//push move to the moves stack
cmove *Engine::create_move(bitboard f, bitboard t, byte moved, byte promto = 0,
		byte flags = 0, bitboard mov2 = 0) {
	bitboard atkmoves = 0;
	byte cap = 0, incheck = 0;

//get captured piece
	for (int i = 1; i < 6; i++) {
		if (t & pieces[movenotof][i]) {
			cap = i;
			break;
		}
	}
// create cmove object
	cmove *m = cmove::newcmove(f, t, mov2, moved, promto, flags, cap);

//if in check reject the move
	domove(m);
	gen_atk_moves(/*moveof, */atkmoves);
//	gen_atk_moves(moveof, atkmoves);
	if (pieces[movenotof][king] & atkmoves)
		incheck = 1;
//check castling
	else if (flags & FLAGCASTLEA) {
		if (moveof == black && (0x18 & atkmoves))
			incheck = 1;
		else if (moveof == white && (0x1800000000000000ULL & atkmoves))
			incheck = 1;
	} else if (flags & FLAGCASTLEH) {
		if (moveof == black && (0x30 & atkmoves))
			incheck = 1;
		else if (moveof == white && (0x3000000000000000ULL & atkmoves))
			incheck = 1;
	}
	/*if (!incheck)
	 m->score = evaluate();*/
	undomove(m);
	if (incheck) {
		cmove::deletecmove(m);
		return NULL;
	}
	return m;
}

//push move to the moves stack
cmove *Engine::create_move_nc(bitboard f, bitboard t, byte moved, byte promto =
		0, byte flags = 0, bitboard mov2 = 0, int weight = 0) {
	byte cap = 0;

//get captured piece
	for (int i = 1; i < 6; i++) {
		if (t & pieces[movenotof][i]) {
			cap = i;
			break;
		}
	}
// create cmove object
	cmove *m = cmove::newcmove(f, t, mov2, moved, promto, flags, cap);
	m->weight = weight;

	return m;
}

//generate moves for a given pawn position
void Engine::gen_pawn_moves(MoveNode *par) {
// squares not occupied by any piece
	bitboard emptysq = ~(all[white] | all[black]);
//iterate thru' all the pawns
	bitboard ap = pieces[moveof][pawn]; //all pawns
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
					par->addChild(create_move(lsb, m, pawn, queen)); //queen
					par->addChild(create_move(lsb, m, pawn, rook)); //rook
					par->addChild(create_move(lsb, m, pawn, bishop)); //bishop
					par->addChild(create_move(lsb, m, pawn, knight)); //knight
				} else
					par->addChild(create_move(lsb, m, pawn));
				//now for the double move
				//only pawns at rank 2 can do that
				if (lsb & rank2) {
					//one more rank forward
					m <<= 8;
					if (m & emptysq)
						//push it in...
						//this is a double move, flag it also
						par->addChild(
								create_move(lsb, m, pawn, 0, FLAGDOUBLEMOVE));
				}
			}
			//lets checkout capture squares
			//capturing on right?
			c = lsb << 9;
			//make sure it doesnt hit the wall
			c &= ~filea;
			//is this en passant
			if (c & epsq)
				par->addChild(create_move(lsb, c, pawn, 0, FLAGENPASSANT));
			//make sure it captures adversary only
			c &= all[black];
			//if capture exists then push it
			if (c) {
				if (c & rank8) //pawn is promoted to...
						{
					par->addChild(create_move(lsb, c, pawn, queen)); //queen
					par->addChild(create_move(lsb, c, pawn, rook)); //rook
					par->addChild(create_move(lsb, c, pawn, bishop)); //bishop
					par->addChild(create_move(lsb, c, pawn, knight)); //knight
				} else
					par->addChild(create_move(lsb, c, pawn));
			}
			//capture on left!
			c = lsb << 7;
			c &= ~fileh;
			if (c & epsq)
				par->addChild(create_move(lsb, c, pawn, 0, FLAGENPASSANT));
			c &= all[black];
			if (c) {
				if (c & rank8) //pawn is promoted to...
						{
					par->addChild(create_move(lsb, c, pawn, queen)); //queen
					par->addChild(create_move(lsb, c, pawn, rook)); //rook
					par->addChild(create_move(lsb, c, pawn, bishop)); //bishop
					par->addChild(create_move(lsb, c, pawn, knight)); //knight
				} else
					par->addChild(create_move(lsb, c, pawn));
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
					par->addChild(create_move(lsb, m, pawn, queen)); //queen
					par->addChild(create_move(lsb, m, pawn, rook)); //rook
					par->addChild(create_move(lsb, m, pawn, bishop)); //bishop
					par->addChild(create_move(lsb, m, pawn, knight)); //knight
				} else
					par->addChild(create_move(lsb, m, pawn));
				//double move! for black it will be from rank 7.
				if (lsb & rank7) {
					m >>= 8;
					if (m & emptysq)
						par->addChild(
								create_move(lsb, m, pawn, 0, FLAGDOUBLEMOVE));
				}
			}
			//capture moves for black
			c = lsb >> 9;
			c &= ~fileh;
			if (c & epsq)
				par->addChild(create_move(lsb, c, pawn, 0, FLAGENPASSANT));
			c &= all[white];
			if (c) {
				if (c & rank1) //pawn is promoted to...
						{
					par->addChild(create_move(lsb, c, pawn, queen)); //queen
					par->addChild(create_move(lsb, c, pawn, rook)); //rook
					par->addChild(create_move(lsb, c, pawn, bishop)); //bishop
					par->addChild(create_move(lsb, c, pawn, knight)); //knight
				} else
					par->addChild(create_move(lsb, c, pawn));
			}
			c = lsb >> 7;
			c &= ~filea;
			if (c & epsq)
				par->addChild(create_move(lsb, c, pawn, 0, FLAGENPASSANT));
			c &= all[white];
			if (c) {
				if (c & rank1) //pawn is promoted to...
						{
					par->addChild(create_move(lsb, c, pawn, queen)); //queen
					par->addChild(create_move(lsb, c, pawn, rook)); //rook
					par->addChild(create_move(lsb, c, pawn, bishop)); //bishop
					par->addChild(create_move(lsb, c, pawn, knight)); //knight
				} else
					par->addChild(create_move(lsb, c, pawn));
			}
		}
		//remove the lsb from pawns bits
		ap ^= lsb;
	}
}

//generate moves for a given pawn position
void Engine::gen_pawn_pseudo_moves(vector<cmove*>&v) {
// squares not occupied by any piece
	bitboard emptysq = ~(all[white] | all[black]);
//iterate thru' all the pawns
	bitboard ap = pieces[moveof][pawn]; //all pawns
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
					v.push_back(create_move_nc(lsb, m, pawn, queen)); //queen
					v.push_back(create_move_nc(lsb, m, pawn, rook)); //rook
					v.push_back(create_move_nc(lsb, m, pawn, bishop)); //bishop
					v.push_back(create_move_nc(lsb, m, pawn, knight)); //knight
				} else
					v.push_back(create_move_nc(lsb, m, pawn));
				//now for the double move
				//only pawns at rank 2 can do that
				if (lsb & rank2) {
					//one more rank forward
					m <<= 8;
					if (m & emptysq)
						//push it in...
						//this is a double move, flag it also
						v.push_back(
								create_move_nc(lsb, m, pawn, 0,
										FLAGDOUBLEMOVE));
				}
			}
			//lets checkout capture squares
			//capturing on right?
			c = lsb << 9;
			//make sure it doesnt hit the wall
			c &= ~filea;
			//is this en passant
			if (c & epsq)
				v.push_back(
						create_move_nc(lsb, c, pawn, 0, FLAGENPASSANT, 0, 800));
			//make sure it captures adversary only
			c &= all[black];
			//if capture exists then push it
			if (c) {
				if (c & rank8) //pawn is promoted to...
						{
					v.push_back(create_move_nc(lsb, c, pawn, queen)); //queen
					v.push_back(create_move_nc(lsb, c, pawn, rook)); //rook
					v.push_back(create_move_nc(lsb, c, pawn, bishop)); //bishop
					v.push_back(create_move_nc(lsb, c, pawn, knight)); //knight
				} else
					v.push_back(create_move_nc(lsb, c, pawn, 0, 0, 0, 800));
			}
			//capture on left!
			c = lsb << 7;
			c &= ~fileh;
			if (c & epsq)
				v.push_back(
						create_move_nc(lsb, c, pawn, 0, FLAGENPASSANT, 0, 800));
			c &= all[black];
			if (c) {
				if (c & rank8) //pawn is promoted to...
						{
					v.push_back(create_move_nc(lsb, c, pawn, queen)); //queen
					v.push_back(create_move_nc(lsb, c, pawn, rook)); //rook
					v.push_back(create_move_nc(lsb, c, pawn, bishop)); //bishop
					v.push_back(create_move_nc(lsb, c, pawn, knight)); //knight
				} else
					v.push_back(create_move_nc(lsb, c, pawn, 0, 0, 0, 800));
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
					v.push_back(create_move_nc(lsb, m, pawn, queen)); //queen
					v.push_back(create_move_nc(lsb, m, pawn, rook)); //rook
					v.push_back(create_move_nc(lsb, m, pawn, bishop)); //bishop
					v.push_back(create_move_nc(lsb, m, pawn, knight)); //knight
				} else
					v.push_back(create_move_nc(lsb, m, pawn));
				//double move! for black it will be from rank 7.
				if (lsb & rank7) {
					m >>= 8;
					if (m & emptysq)
						v.push_back(
								create_move_nc(lsb, m, pawn, 0,
										FLAGDOUBLEMOVE));
				}
			}
			//capture moves for black
			c = lsb >> 9;
			c &= ~fileh;
			if (c & epsq)
				v.push_back(
						create_move_nc(lsb, c, pawn, 0, FLAGENPASSANT, 0, 800));
			c &= all[white];
			if (c) {
				if (c & rank1) //pawn is promoted to...
						{
					v.push_back(create_move_nc(lsb, c, pawn, queen)); //queen
					v.push_back(create_move_nc(lsb, c, pawn, rook)); //rook
					v.push_back(create_move_nc(lsb, c, pawn, bishop)); //bishop
					v.push_back(create_move_nc(lsb, c, pawn, knight)); //knight
				} else
					v.push_back(create_move_nc(lsb, c, pawn, 0, 0, 0, 800));
			}
			c = lsb >> 7;
			c &= ~filea;
			if (c & epsq)
				v.push_back(
						create_move_nc(lsb, c, pawn, 0, FLAGENPASSANT, 0, 800));
			c &= all[white];
			if (c) {
				if (c & rank1) //pawn is promoted to...
						{
					v.push_back(create_move_nc(lsb, c, pawn, queen)); //queen
					v.push_back(create_move_nc(lsb, c, pawn, rook)); //rook
					v.push_back(create_move_nc(lsb, c, pawn, bishop)); //bishop
					v.push_back(create_move_nc(lsb, c, pawn, knight)); //knight
				} else
					v.push_back(create_move_nc(lsb, c, pawn, 0, 0, 0, 800));
			}
		}
		//remove the lsb from pawns bits
		ap ^= lsb;
	}
}

//generate pawn captures
void Engine::gen_pawn_captures(vector<cmove*>&v) {
//iterate thru' all the pawns
	bitboard ap = pieces[moveof][pawn]; //all pawns
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
		if (moveof == white) {
			//lets checkout capture squares
			//capturing on right?
			c = lsb << 9;
			//make sure it doesnt hit the wall
			c &= ~filea;
			//is this en passant
			if (c & epsq)
				v.push_back(
						create_move_nc(lsb, c, pawn, 0, FLAGENPASSANT, 0, 800));
			//make sure it captures adversary only
			c &= all[black];
			//if capture exists then push it
			if (c) {
				if (c & rank8) //pawn is promoted to...
						{
					v.push_back(create_move_nc(lsb, c, pawn, queen)); //queen
					v.push_back(create_move_nc(lsb, c, pawn, rook)); //rook
					v.push_back(create_move_nc(lsb, c, pawn, bishop)); //bishop
					v.push_back(create_move_nc(lsb, c, pawn, knight)); //knight
				} else
					v.push_back(create_move_nc(lsb, c, pawn, 0, 0, 0, 800));
			}
			//capture on left!
			c = lsb << 7;
			c &= ~fileh;
			if (c & epsq)
				v.push_back(
						create_move_nc(lsb, c, pawn, 0, FLAGENPASSANT, 0, 800));
			c &= all[black];
			if (c) {
				if (c & rank8) //pawn is promoted to...
						{
					v.push_back(create_move_nc(lsb, c, pawn, queen)); //queen
					v.push_back(create_move_nc(lsb, c, pawn, rook)); //rook
					v.push_back(create_move_nc(lsb, c, pawn, bishop)); //bishop
					v.push_back(create_move_nc(lsb, c, pawn, knight)); //knight
				} else
					v.push_back(create_move_nc(lsb, c, pawn, 0, 0, 0, 800));
			}

		}
		//same goes for the black pawns
		//just for the fact that they move in opposite direction.
		else {
			//capture moves for black
			c = lsb >> 9;
			c &= ~fileh;
			if (c & epsq)
				v.push_back(
						create_move_nc(lsb, c, pawn, 0, FLAGENPASSANT, 0, 800));
			c &= all[white];
			if (c) {
				if (c & rank1) //pawn is promoted to...
						{
					v.push_back(create_move_nc(lsb, c, pawn, queen)); //queen
					v.push_back(create_move_nc(lsb, c, pawn, rook)); //rook
					v.push_back(create_move_nc(lsb, c, pawn, bishop)); //bishop
					v.push_back(create_move_nc(lsb, c, pawn, knight)); //knight
				} else
					v.push_back(create_move_nc(lsb, c, pawn, 0, 0, 0, 800));
			}
			c = lsb >> 7;
			c &= ~filea;
			if (c & epsq)
				v.push_back(
						create_move_nc(lsb, c, pawn, 0, FLAGENPASSANT, 0, 800));
			c &= all[white];
			if (c) {
				if (c & rank1) //pawn is promoted to...
						{
					v.push_back(create_move_nc(lsb, c, pawn, queen)); //queen
					v.push_back(create_move_nc(lsb, c, pawn, rook)); //rook
					v.push_back(create_move_nc(lsb, c, pawn, bishop)); //bishop
					v.push_back(create_move_nc(lsb, c, pawn, knight)); //knight
				} else
					v.push_back(create_move_nc(lsb, c, pawn, 0, 0, 0, 800));
			}
		}
		//remove the lsb from pawns bits
		ap ^= lsb;
	}
}

//generate attack moves for pawn
void Engine::gen_pawn_atk(/*side moveof, */bitboard& atkbrd) {
//void Engine::gen_pawn_atk(side mof, bitboard& atkbrd) {
//iterate thru' all the pawns
	side mof = moveof;
	bitboard ap = pieces[mof][pawn]; //all pawns
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
		if (mof == white) {
			//lets checkout capture squares
			//capturing on right?
			c = lsb << 9;
			//make sure it doesnt hit the wall
			c &= ~filea;
			//make sure it captures adversary only
			c &= all[black];
			//record capture
			atkbrd |= c;
			//capture on left!
			c = lsb << 7;
			c &= ~fileh;
			c &= all[black];
			atkbrd |= c;
		}
		//same goes for the black pawns
		//just for the fact that they move in opposite direction.
		else {
			//capture moves for black
			c = lsb >> 9;
			c &= ~fileh;
			c &= all[white];
			atkbrd |= c;

			c = lsb >> 7;
			c &= ~filea;
			c &= all[white];
			atkbrd |= c;
		}
		//remove the lsb from pawns bits
		ap ^= lsb;
	}
}
// generate king moves
void Engine::gen_king_moves(MoveNode *par) {
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof], _all = all[white] | all[black];
//get king position
	bitboard lsb = pieces[moveof][king];
	bitboard m; //moved to position
	int lsbint;
	bitboard ant/*all move to positions*/;
	lsbint = get_bit_pos(lsb);
// get all moves on that position
	ant = king_moves[lsbint];

// generate castling move if allowed
	if (!kingmoved[moveof]) {
		if (!isUnderAttack(pieces[king][moveof], movenotof, moveof)) {
			// castle towards filea
			if (!rookmoved[moveof][0]) {
				if (moveof == white && !(_all & (b1 | c1 | d1))
						&& (pieces[white][rook] & a1)) {
					par->addChild(
							create_move(lsb, c1, king, 0,
									FLAGCASTLEA | KINGMOVED, a1 | d1));
				} else if (moveof == black && !(_all & (b8 | c8 | d8))
						&& (pieces[black][rook] & a8)) {
					par->addChild(
							create_move(lsb, c8, king, 0,
									FLAGCASTLEA | KINGMOVED, a8 | d8));
				}
			}
			// castle towards fileh
			if (!rookmoved[moveof][1]) {
				if (moveof == white && !(_all & (f1 | g1))
						&& (pieces[white][rook] & h1)) {
					par->addChild(
							create_move(lsb, g1, king, 0,
									FLAGCASTLEH | KINGMOVED, h1 | f1));
				} else if (moveof == black && !(_all & (f8 | g8))
						&& (pieces[black][rook] & h8)) {
					par->addChild(
							create_move(lsb, g8, king, 0,
									FLAGCASTLEH | KINGMOVED, h8 | f8));
				}
			}
		}
	}
// loop thru' all moves
	while (ant) {
		m = ant & (~ant + 1);
		if (m & othersq) {
			par->addChild(create_move(lsb, m, king, 0, KINGMOVED));
		}
		ant ^= m;
	}
}
// generate king moves
void Engine::gen_king_pseudo_moves(vector<cmove*>&v) {
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof], _all = all[white] | all[black];
//get king position
	bitboard lsb = pieces[moveof][king];
	bitboard m; //moved to position
	int lsbint;
	bitboard ant/*all move to positions*/;
	lsbint = get_bit_pos(lsb);
// get all moves on that position
	ant = king_moves[lsbint];

// generate castling move if allowed
	if (!kingmoved[moveof]) {
		if (!isUnderAttack(pieces[king][moveof], movenotof, moveof)) {
			// castle towards filea
			if (!rookmoved[moveof][0]) {
				if (moveof == white && !(_all & (b1 | c1 | d1))
						&& (pieces[white][rook] & a1)) {
					v.push_back(
							create_move_nc(lsb, c1, king, 0,
									FLAGCASTLEA | KINGMOVED, a1 | d1));
				} else if (moveof == black && !(_all & (b8 | c8 | d8))
						&& (pieces[black][rook] & a8)) {
					v.push_back(
							create_move_nc(lsb, c8, king, 0,
									FLAGCASTLEA | KINGMOVED, a8 | d8));
				}
			}
			// castle towards fileh
			if (!rookmoved[moveof][1]) {
				if (moveof == white && !(_all & (f1 | g1))
						&& (pieces[white][rook] & h1)) {
					v.push_back(
							create_move_nc(lsb, g1, king, 0,
									FLAGCASTLEH | KINGMOVED, h1 | f1));
				} else if (moveof == black && !(_all & (f8 | g8))
						&& (pieces[black][rook] & h8)) {
					v.push_back(
							create_move_nc(lsb, g8, king, 0,
									FLAGCASTLEH | KINGMOVED, h8 | f8));
				}
			}
		}
	}
// loop thru' all moves
	while (ant) {
		m = ant & (~ant + 1);

		if (m & all[movenotof])
			v.push_back(create_move_nc(lsb, m, king, 0, 0, 0, 400));
		else if (m & othersq) {
			v.push_back(create_move_nc(lsb, m, king, 0, KINGMOVED));
		}
		ant ^= m;
	}
}

// generate king captures
void Engine::gen_king_captures(vector<cmove*>&v) {
//get king position
	bitboard lsb = pieces[moveof][king];
	bitboard m; //moved to position
	int lsbint;
	bitboard ant/*all move to positions*/;
	lsbint = get_bit_pos(lsb);
// get all moves on that position
	ant = king_moves[lsbint] & all[movenotof];

// loop thru' all moves
	while (ant) {
		m = ant & (~ant + 1);
		v.push_back(create_move_nc(lsb, m, king, 0, 0, 0, 400));
		ant ^= m;
	}
}

// generate king attack moves
void Engine::gen_king_atk(/*side moveof, */bitboard& atkbrd) {
//void Engine::gen_king_atk(side mof, bitboard& atkbrd) {
//get king position
	bitboard lsb = pieces[moveof][king];
	int lsbint = get_bit_pos(lsb);
// get all moves on that position
	atkbrd |= king_moves[lsbint];
}
// generate knight moves
void Engine::gen_knight_moves(MoveNode *par) {
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
//get position
	bitboard an = pieces[moveof][knight];
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
			if (m & othersq) {
				par->addChild(create_move(lsb, m, knight));
			}
			ant ^= m;
		}
		//remove the lsb from pawns bits
		an ^= lsb;
	}
}
// generate knight moves
void Engine::gen_knight_pseudo_moves(vector<cmove*>&v) {
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
//get position
	bitboard an = pieces[moveof][knight];
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
			if (m & all[movenotof])
				v.push_back(create_move_nc(lsb, m, knight, 0, 0, 0, 700));
			else if (m & othersq) {
				v.push_back(create_move_nc(lsb, m, knight));
			}
			ant ^= m;
		}
		//remove the lsb from pawns bits
		an ^= lsb;
	}
}

// generate knight captures
void Engine::gen_knight_captures(vector<cmove*>&v) {
//get position
	bitboard an = pieces[moveof][knight];
	bitboard lsb; //last significant bit
	bitboard m; //moved to position
	int lsbint;
	bitboard ant/*all move to positions*/;
//iterate thru' all knights
	while (an) {
		lsb = an & (~an + 1);
		lsbint = get_bit_pos(lsb);
		// get all knight moves on that position
		ant = knight_moves[lsbint] & all[movenotof];
		// loop thru' all moves
		while (ant) {
			m = ant & (~ant + 1);
			v.push_back(create_move_nc(lsb, m, knight, 0, 0, 0, 700));
			ant ^= m;
		}
		//remove the lsb from pawns bits
		an ^= lsb;
	}
}

// generate knight attack moves
void Engine::gen_knight_atk(/*side moveof, */bitboard& atkbrd) {
//void Engine::gen_knight_atk(side mof, bitboard& atkbrd) {
//get position
	bitboard an = pieces[moveof][knight];
	bitboard lsb; //last significant bit
	int lsbint;
	bitboard ant/*all move to positions*/;
//iterate thru' all knights
	while (an) {
		//lsb = an & (~an + 1);
		lsb = an & -an;
		lsbint = get_bit_pos(lsb);
		// get all knight moves on that position
		ant = knight_moves[lsbint];
		// loop thru' all moves
		atkbrd |= ant;
		//remove the lsb from pawns bits
		an ^= lsb;
	}
}
// generate rook moves
void Engine::gen_rook_moves(byte piecefor, bitboard ar, MoveNode *par) {
	byte flag = 0;
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

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
				if (piecefor == rook) {
					if ((lsb & filea) && rookmoved[moveof][0] == 0)
						flag = ROOKMOVED;
					else if ((lsb & fileh) && rookmoved[moveof][1] == 0)
						flag = ROOKMOVED;
					else
						flag = 0;
				} else
					flag = 0;
				par->addChild(create_move(lsb, m, piecefor, 0, flag));
			}
			_dm ^= m;
		}
		ar ^= lsb;
	}
}
// generate rook moves
void Engine::gen_rook_pseudo_moves(byte piecefor, bitboard ar,
		vector<cmove*>&v) {
	byte flag = 0;
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

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
				if (piecefor == rook) {
					if ((lsb & filea) && rookmoved[moveof][0] == 0)
						flag = ROOKMOVED;
					else if ((lsb & fileh) && rookmoved[moveof][1] == 0)
						flag = ROOKMOVED;
					else
						flag = 0;
				} else
					flag = 0;

				if (m & all[movenotof])
					v.push_back(
							create_move_nc(lsb, m, piecefor, 0, flag, 0, 500));
				else
					v.push_back(create_move_nc(lsb, m, piecefor, 0, flag));
			}
			_dm ^= m;
		}
		ar ^= lsb;
	}
}

// generate rook captures
void Engine::gen_rook_captures(byte piecefor, bitboard ar, vector<cmove*>&v) {
	byte flag = 0;
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

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
		_dm &= all[movenotof];
		while (_dm) {
			m = _dm & (~_dm + 1);
			if (m & othersq) {
				if (piecefor == rook) {
					if ((lsb & filea) && rookmoved[moveof][0] == 0)
						flag = ROOKMOVED;
					else if ((lsb & fileh) && rookmoved[moveof][1] == 0)
						flag = ROOKMOVED;
				}
				v.push_back(create_move_nc(lsb, m, piecefor, 0, flag, 0, 500));
			}
			_dm ^= m;
		}
		ar ^= lsb;
	}
}

// generate rook moves
bitboard Engine::gen_rook_moves2(bitboard rp, side moveof) {
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

	int lsbint;
	bitboard _rm, _lm, _um, _dm;
	lsbint = get_bit_pos(rp);
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
	return _dm | _um | _lm | _rm;
}
// generate rook attack moves
void Engine::gen_rook_atk(/*side moveof, */bitboard& atkbrd) {
//void Engine::gen_rook_atk(side mof, bitboard& atkbrd) {
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

	bitboard lsb; //last significant bit
	int lsbint;
	bitboard _rm, _lm, _um, _dm;
//iterate thru' all rooks/queens
	bitboard ar = pieces[moveof][rook] | pieces[moveof][queen];
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

bitboard Engine::king_atk_brd(int lsbint, side gf) {
	// get all moves on that position
	return king_moves[lsbint];
}

bitboard Engine::bishop_atk_brd(int lsbint, side gf) {
	// squares not occupied by our pieces
	bitboard othersq = ~all[gf];
	// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

	bitboard _45m, _225m, _135m, _315m;

	//generate moves for diagonally right up
	_45m = deg45_moves[lsbint] & _all;
	_45m = _45m << 9 | _45m << 18 | _45m << 27 | _45m << 36 | _45m << 45
			| _45m << 54;
	_45m &= deg45_moves[lsbint];
	_45m ^= deg45_moves[lsbint];
	_45m &= othersq;

	// generate moves for left down
	_225m = deg225_moves[lsbint] & _all;
	_225m = _225m >> 9 | _225m >> 18 | _225m >> 27 | _225m >> 36 | _225m >> 45
			| _225m >> 54;
	_225m &= deg225_moves[lsbint];
	_225m ^= deg225_moves[lsbint];
	_225m &= othersq;

	// generate moves right down
	_135m = deg135_moves[lsbint] & _all;
	_135m = _135m >> 7 | _135m >> 14 | _135m >> 21 | _135m >> 28 | _135m >> 35
			| _135m >> 42;
	_135m &= deg135_moves[lsbint];
	_135m ^= deg135_moves[lsbint];
	_135m &= othersq;

	// generate moves for left up
	_315m = deg315_moves[lsbint] & _all;
	_315m = _315m << 7 | _315m << 14 | _315m << 21 | _315m << 28 | _315m << 35
			| _315m << 42;
	_315m &= deg315_moves[lsbint];
	_315m ^= deg315_moves[lsbint];
	_315m &= othersq;

	// loop thru' all moves
	_315m = _315m | _135m | _225m | _45m;
	return _315m;
}

bitboard Engine::knight_atk_brd(int lsbint, side gf) {
	// get all knight moves on that position
	return knight_moves[lsbint];
}

bitboard Engine::pawn_atk_brd(bitboard lsb, side gf) {
	bitboard atkbrd = 0;
	bitboard c = 0;
	//if they are white
	if (gf == white) {
		//lets checkout capture squares
		//capturing on right?
		c = lsb << 9;
		//make sure it doesnt hit the wall
		c &= ~filea;
		//make sure it captures adversary only
		c &= all[black];
		//record capture
		atkbrd |= c;
		//capture on left!
		c = lsb << 7;
		c &= ~fileh;
		c &= all[black];
		atkbrd |= c;
	}
	//same goes for the black pawns
	//just for the fact that they move in opposite direction.
	else {
		//capture moves for black
		c = lsb >> 9;
		c &= ~fileh;
		c &= all[white];
		atkbrd |= c;

		c = lsb >> 7;
		c &= ~filea;
		c &= all[white];
		atkbrd |= c;
	}
	return atkbrd;
}

// generate rook attack moves
bitboard Engine::rook_atk_brd(int lsbint, side gf) {
//void Engine::gen_rook_atk(side mof, bitboard& atkbrd) {
// squares not occupied by our pieces
	bitboard othersq = ~all[gf];
// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

	bitboard _rm, _lm, _um, _dm;
//iterate thru' all rooks/queens
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
	return _dm;
}

// generate bishop moves
void Engine::gen_bishop_moves(byte piecefor, bitboard ab, MoveNode *par) {
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

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
			//if (m & othersq) {
			par->addChild(create_move(lsb, m, piecefor));
			//}
			_315m ^= m;
		}
		ab ^= lsb;
	}
}
// generate bishop moves
void Engine::gen_bishop_pseudo_moves(byte piecefor, bitboard ab,
		vector<cmove*>&v) {
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

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
			if (m & all[movenotof])
				v.push_back(create_move_nc(lsb, m, piecefor, 0, 0, 0, 600));
			else
				v.push_back(create_move_nc(lsb, m, piecefor));
			_315m ^= m;
		}
		ab ^= lsb;
	}
}

// generate bishop captures
void Engine::gen_bishop_captures(byte piecefor, bitboard ab, vector<cmove*>&v) {
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

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
		_315m &= all[movenotof];
		while (_315m) {
			m = _315m & (~_315m + 1);
			v.push_back(create_move_nc(lsb, m, piecefor, 0, 0, 0, 600));
			_315m ^= m;
		}
		ab ^= lsb;
	}
}

// generate bishop moves
bitboard Engine::gen_bishop_moves2(bitboard bp, side moveof) {
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

	int lsbint;
	bitboard _45m, _225m, _135m, _315m;

	lsbint = get_bit_pos(bp);
//generate moves for diagonally right up
	_45m = deg45_moves[lsbint] & _all;
	_45m = _45m << 9 | _45m << 18 | _45m << 27 | _45m << 36 | _45m << 45
			| _45m << 54;
	_45m &= deg45_moves[lsbint];
	_45m ^= deg45_moves[lsbint];
	_45m &= othersq;

// generate moves for left down
	_225m = deg225_moves[lsbint] & _all;
	_225m = _225m >> 9 | _225m >> 18 | _225m >> 27 | _225m >> 36 | _225m >> 45
			| _225m >> 54;
	_225m &= deg225_moves[lsbint];
	_225m ^= deg225_moves[lsbint];
	_225m &= othersq;

// generate moves right down
	_135m = deg135_moves[lsbint] & _all;
	_135m = _135m >> 7 | _135m >> 14 | _135m >> 21 | _135m >> 28 | _135m >> 35
			| _135m >> 42;
	_135m &= deg135_moves[lsbint];
	_135m ^= deg135_moves[lsbint];
	_135m &= othersq;

// generate moves for left up
	_315m = deg315_moves[lsbint] & _all;
	_315m = _315m << 7 | _315m << 14 | _315m << 21 | _315m << 28 | _315m << 35
			| _315m << 42;
	_315m &= deg315_moves[lsbint];
	_315m ^= deg315_moves[lsbint];
	_315m &= othersq;

// loop thru' all moves
	return _315m | _135m | _225m | _45m;
}
// generate bishop attack moves
void Engine::gen_bishop_atk(/*side moveof, */bitboard& atkbrd) {
//void Engine::gen_bishop_atk(side mof, bitboard& atkbrd) {
// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

	bitboard lsb; //last significant bit
	int lsbint;
	bitboard _45m, _225m, _135m, _315m;
//iterate thru' all bishops/queens
	bitboard ab = pieces[moveof][bishop] | pieces[moveof][queen];
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
//generate moves for the given on the current board position
void Engine::generate_moves(MoveNode *par) {
	gen_king_moves(par);
// rook and bishop moves for the queen
	gen_rook_moves(queen, pieces[moveof][queen], par);
	gen_bishop_moves(queen, pieces[moveof][queen], par);
// actual rook and bishop moves
	gen_rook_moves(rook, pieces[moveof][rook], par);
	gen_bishop_moves(bishop, pieces[moveof][bishop], par);
	gen_knight_moves(par);
	gen_pawn_moves(par);
}

//generate pseudo-legal moves for the given on the current board position
void Engine::generate_pseudo_moves(vector<cmove*>&v) {
	gen_king_pseudo_moves(v);
// rook and bishop moves for the queen
	gen_rook_pseudo_moves(queen, pieces[moveof][queen], v);
	gen_bishop_pseudo_moves(queen, pieces[moveof][queen], v);
// actual rook and bishop moves
	gen_rook_pseudo_moves(rook, pieces[moveof][rook], v);
	gen_bishop_pseudo_moves(bishop, pieces[moveof][bishop], v);
	gen_knight_pseudo_moves(v);
	gen_pawn_pseudo_moves(v);
}

//generate pseudo-legal moves for the given on the current board position
void Engine::generate_captures(vector<cmove*>&v) {
	gen_king_captures(v);
// rook and bishop moves for the queen
	gen_rook_captures(queen, pieces[moveof][queen], v);
	gen_bishop_captures(queen, pieces[moveof][queen], v);
// actual rook and bishop moves
	gen_rook_captures(rook, pieces[moveof][rook], v);
	gen_bishop_captures(bishop, pieces[moveof][bishop], v);
	gen_knight_captures(v);
	gen_pawn_captures(v);
}

//generate attack moves
void Engine::gen_atk_moves(/*side moveof, */bitboard& atkbrd) {
//void Engine::gen_atk_moves(side mof, bitboard& atkbrd) {
	gen_king_atk(atkbrd);
	gen_pawn_atk(atkbrd);
	gen_knight_atk(atkbrd);
	gen_rook_atk(atkbrd);
	gen_bishop_atk(atkbrd);
	/*gen_king_atk(mof, atkbrd);
	 gen_pawn_atk(mof, atkbrd);
	 gen_knight_atk(mof, atkbrd);
	 gen_rook_atk(mof, atkbrd);
	 gen_bishop_atk(mof, atkbrd);*/
}
/*
 * check_move ()
 * check if user/xboard move is valid or not
 * ----
 * if from is not valid return 0
 * if to is not valid return 0
 * if piece move not valid return 0
 * do move
 * if in check undo move and return 0
 * return 1
 */
cmove *Engine::check_move(bitboard f, bitboard t, int promto = 0) {
	cmove *m = NULL;
	bitboard atkmoves = 0;
	int cap = 0, incheck = 0;

//if from is not valid return 0:"<<f<<":"<<moveof<<"\n";
	if (!(f & all[moveof])) {
		cout << "from not valid\n";
		return NULL;
	}
//if to is not valid return 0\n";
	if (!(t & ~all[moveof]))
		return NULL;
//get captured piece\n";
	for (int i = 1; i < 6; i++) {
		if (t & pieces[movenotof][i]) {
			cap = i;
			break;
		}
	}
//if piece move not valid return 0\n";
	m = check_piece_move(f, t, promto, cap);
	if (!m)
		return NULL;

	domove(m);
//if in check undo move and return 0\n";
	gen_atk_moves(/*moveof, */atkmoves);
	//gen_atk_moves(moveof, atkmoves);
	if (pieces[movenotof][king] & atkmoves)
		incheck = 1;
	undomove(m);
	if (incheck)
		return NULL;
	return m;
}

cmove *Engine::check_piece_move(bitboard f, bitboard t, int promto, int cap) {
// squares not occupied by any piece
	bitboard emptysq = ~(all[white] | all[black]);
	bitboard m;
//if pawn move
	if (f & pieces[moveof][pawn]) {
		// for white
		// forward movement
		if ((f << 8) & emptysq) {
			if (t & (f << 8))
				return cmove::newcmove(f, t, 0, pawn, promto);
			else if (t & (f << 16) & emptysq & rank4)
				return cmove::newcmove(f, t, 0, pawn, 0, FLAGDOUBLEMOVE);
		}
		//capture
		m = f << 9;
		m &= ~filea;
		if (t & m & epsq)
			return cmove::newcmove(f, t, 0, pawn, 0, FLAGENPASSANT);
		if (t & m & all[black])
			return cmove::newcmove(f, t, 0, pawn, promto, 0, cap);
		m = f << 7;
		m &= ~fileh;
		if (t & m & epsq)
			return cmove::newcmove(f, t, 0, pawn, 0, FLAGENPASSANT);
		if (t & m & all[black])
			return cmove::newcmove(f, t, 0, pawn, promto, 0, cap);

		//for black
		if ((f >> 8) & emptysq) {
			if (t & (f >> 8))
				return cmove::newcmove(f, t, 0, pawn, promto);
			else if (t & (f >> 16) & emptysq & rank5)
				return cmove::newcmove(f, t, 0, pawn, 0, FLAGDOUBLEMOVE);
		}
		//capture
		m = f >> 9;
		m &= ~fileh;
		if (t & m & epsq)
			return cmove::newcmove(f, t, 0, pawn, 0, FLAGENPASSANT);
		if (t & m & all[white])
			return cmove::newcmove(f, t, 0, pawn, promto, 0, cap);
		m = f >> 7;
		m &= ~filea;
		if (t & m & epsq)
			return cmove::newcmove(f, t, 0, pawn, 0, FLAGENPASSANT);
		if (t & m & all[white])
			return cmove::newcmove(f, t, 0, pawn, promto, 0, cap);
	} else if (f & pieces[moveof][queen]) {
		if ((t & gen_rook_moves2(f, moveof))
				|| (t & gen_bishop_moves2(f, moveof)))
			return cmove::newcmove(f, t, 0, queen, 0, 0, cap);
	} else if (f & pieces[moveof][rook]) {
		if (t & gen_rook_moves2(f, moveof))
			return cmove::newcmove(f, t, 0, rook, 0, ROOKMOVED, cap);
	} else if (f & pieces[moveof][bishop]) {
		if (t & gen_bishop_moves2(f, moveof))
			return cmove::newcmove(f, t, 0, bishop, 0, 0, cap);
	} else if (f & pieces[moveof][knight]) {
		int intf = get_bit_pos(f);
		m = knight_moves[intf] & ~all[moveof];
		if (t & m)
			return cmove::newcmove(f, t, 0, knight, 0, 0, cap);
	} else if (f & pieces[moveof][king]) {
		int intf = get_bit_pos(f);
		m = king_moves[intf] & ~all[moveof];
		if (t & m)
			return cmove::newcmove(f, t, 0, king, 0, KINGMOVED, cap);

		// generate castling move if allowed
		// need to check for check
		if (!kingmoved[moveof] && (f & start_pieces[moveof][king])) { // if king hasnt moved
			// castle towards filea
			if (!rookmoved[moveof][0] && (t & filec)) { // if file A rook hasnt moved
				if (moveof == white && !((all[white] | all[black]) & 0xE)
						&& (pieces[white][rook] & 0x1)) {
					return cmove::newcmove(f, t, 0x1 | 0x8, king, 0,
							FLAGCASTLEA | KINGMOVED);
				} else if (moveof == black
						&& !((all[white] | all[black]) & 0xE00000000000000ULL)
						&& (pieces[black][rook] & 0x100000000000000ULL)) {
					return cmove::newcmove(f, t,
							0x100000000000000ULL | 0x800000000000000ULL, king,
							0, FLAGCASTLEA | KINGMOVED);
				}
			}
			// castle towards fileh
			if (!rookmoved[moveof][1] && (t & fileg)) {
				if (moveof == white && !((all[white] | all[black]) & 0x60)
						&& (pieces[white][rook] & 0x80)) {
					return cmove::newcmove(f, t, 0x80 | 0x20, king, 0,
							FLAGCASTLEH | KINGMOVED);
				} else if (moveof == black
						&& !((all[white] | all[black]) & 0x6000000000000000ULL)
						&& (pieces[black][rook] & 0x8000000000000000ULL)) {
					return cmove::newcmove(f, t,
							0x8000000000000000ULL | 0x2000000000000000ULL, king,
							0, FLAGCASTLEH | KINGMOVED);
				}
			}
		}
	} else {
		cout << "error@check_piece_move\n";
	}

	return NULL;
}

void Engine::list_moves() {
	MoveNode dummy;
	generate_moves(&dummy);
	for (MoveNode *cur = dummy.child; cur; cur = cur->next)
		cout << cur->move->getMoveTxt() << "; ";
	cout << endl;
}

void Engine::inittime(char *t) {
	string s(t);
	stringstream ss(s);
	string cmd;
	int base, inc;

	ss >> cmd >> movesintime >> base >> inc;
	movesintime *= 2;
	mps = movesintime;
	timeleft = base * 60 * CLOCKS_PER_SEC;
}

void Engine::setowntime(char *t) {
	string s(t);
	stringstream ss(s);
	string cmd;

	ss >> cmd >> timeleft;
	timeleft *= 10;
}

void Engine::perft(int depth) {
	newGame();

	for (int i = 1; i <= depth; i++) {
		int nodes = 0;
		//time it
		time_t ts = clock();
		perftloop(i, nodes);
		time_t te = clock();

		float diff = (te - ts);
		float cps = CLOCKS_PER_SEC;

		float sec = diff / cps;
		cout << "result:\n";
		cout << "processed nodes:" << nodes << " in " << sec
				<< " secs at depth " << i << "." << endl;
	}
}

void Engine::perftloop(int depth, int &nodes) {
	if (!depth) {
		nodes++;
		return;
	}

	cmove *m;
	vector<cmove*> v;
	generate_pseudo_moves(v);

	while (!v.empty()) {
		m = v.back();
		domove(m);
		if (!isUnderAttack(pieces[movenotof][king], moveof, movenotof))
			perftloop(depth - 1, nodes);
		undomove(m);

		v.pop_back();
		delete m;
	}
}
