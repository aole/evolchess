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

#include "engine.h"
#include "debug.h"
#include "windows.h"

using namespace std;

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

//initializes the game to starting position
void Engine::newGame() {
	ismate = 0;
	isthreefoldw = isthreefoldb = 0;
	//stack.init();
	moveshistory.init();
	moveno = 0;
	gameended = 0;
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
	cmove *lm = NULL;
	bitboard t = move->to, f = move->from;
	bitboard mov = f | t;
	//switch move of
	moveof = movenotof;
	movenotof = (moveof == white ? black : white);

	moveshistory.pop();

	if (moveof == white)
		if (isthreefoldw)
			isthreefoldw--;
	if (moveof == black)
		if (isthreefoldb)
			isthreefoldb--;

	//for castling purpose
	if (move->piece == king && kingmoved[moveof] == moveno)
		kingmoved[moveof] = 0;
	if (move->piece == rook && f & filea && rookmoved[moveof][0] == moveno)
		rookmoved[moveof][0] = 0;
	else if (move->piece == rook && f & fileh && rookmoved[moveof][1] == moveno)
		rookmoved[moveof][1] = 0;

	//enpassent
	if (move->flags & FLAGENPASSANT) {
		if (moveof == white) {
			all[movenotof] ^= t >> 8;//the pawn to be captured is actually, one rank back
			pieces[movenotof][pawn] ^= t >> 8;
		} else {
			all[movenotof] ^= t << 8;//the pawn to be captured is actually, one rank back
			pieces[movenotof][pawn] ^= t << 8;
		}
	} else
	//respawn captured piece
	if (move->captured) {
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
void Engine::cleannode(MoveNode *node) {
	if (node->child)
		cleannode(node->child);
	if (node->next)
		cleannode(node->next);
	delete node;
	node = NULL;
}
// initialize static variable
Engine *Engine::curengine = NULL;
// ai thread function
void Engine::findbestmove() {
	time_t t1, t2;
	int best_score;
	MoveNode *dummy_node = new MoveNode;
	MoveNode *bm = new MoveNode;

	if (!curengine) {
		curengine->bestmove = NULL;
		return;
	}
	//start timer
	t1 = clock();

	for (int depth = 2; depth <= MAX_AI_SEARCH_DEPTH; depth++) {
		curengine->prosnodes = 0;
		//Alpha-Beta Pruning
		int alpha = -VALUEINFINITE, beta = VALUEINFINITE;

		best_score = curengine->next_ply_best_score(dummy_node, depth, alpha, beta, bm);
		t2 = clock() - t1;
		if (!dummy_node->child) {
			cout << "elapsed:" << t2 << endl;
			curengine->bestmove = NULL;
			return;
		}
		if (t2 > 5000)
			break;
	}
	if (!bm->child)
		bm->child = dummy_node->child;
	curengine->bestmove = new cmove(bm->child->move);
	curengine->domove(curengine->bestmove);

	if (!bm->child->child)
		curengine->ismate = 1;

	delete bm;
	curengine->cleannode(dummy_node);

	cout << "elapsed:" << t2 << "; prossesed:" << curengine->prosnodes << endl;
}
// ai with Negamax Search
cmove *Engine::doaimove() {
	curengine = this;

	HANDLE th = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)findbestmove, NULL, 0, NULL);
	WaitForSingleObject(th, INFINITE);

	return bestmove;
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
		if (c->score >= cur->score) {
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
//
int Engine::next_ply_best_score(MoveNode *par, int depth, int alpha, int beta,
		MoveNode *bm) {
	int cur_score, best_score = -VALUEINFINITE, newnodes = 0;
	MoveNode *par2;

	depth--;
	if (!par->movesgenerated) {
		generate_moves(par);
		par->movesgenerated = newnodes = 1;
	}

	if (checkfordraw())
		return 0;
	if (!par->child) {
		return best_score * (depth + 1);
	}

	if (!newnodes) {
		par2 = new MoveNode;
		//get static scores for this level and sort them
		for (MoveNode *cur = par->child; cur;) {
			cur = insert_sort(par2, cur);
		}
		par->child = par2->child;
		delete par2;
		par2 = NULL;
	}

	for (MoveNode *cur = par->child; cur; cur = cur->next) {
		domove(cur->move);
		if (!depth) {
			prosnodes++;
			cur_score = static_position_score();
			cur->score = cur_score;
		} else {
			cur_score = -next_ply_best_score(cur, depth, -beta, -alpha, NULL);
			cur->score = cur_score;
		}
		undolastmove();

		if (cur_score > best_score) {
			if (bm)
				bm->child = cur;
			best_score = cur_score;
		}
		if (best_score > alpha)
			alpha = best_score;
		if (alpha >= beta) {
			if (bm)
				bm->child = cur;
			//if (depth==1)
			for (cur = cur->next; cur; cur = cur->next)
				cur->score = -VALUEINFINITE;
			return alpha;
		}
	}
	return best_score;
}
//
int Engine::static_position_score() {
	//calculate board value
	int bv[] = { 0, 0 }, j;
	bitboard lsb;//last significant bit

	for (int i = 0; i < 2; i++) {
		if (i == 0)
			j = 1;
		else
			j = 1;

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
		//check for center pawns
		if (pieces[i][pawn] & 0x1818000000ULL)
			bv[i] += 2;
		if (pieces[i][pawn] & 0x181818180000ULL)
			bv[i]++;

		//check for castling
		if (i == white) {
			if (pieces[i][king] & 0x40 && !(all[i] & 0x80) && pieces[i][pawn]
					& 0xE000) {
				bv[i] += 10;
			}
		} else {
			if (pieces[i][king] & 0x4000000000000000ULL && !(all[i]
					& 0x8000000000000000ULL) && pieces[i][pawn]
					& 0xE000000000000000ULL) {
				bv[i] += 10;
			}
		}
		//knights in the middle
		if (pieces[i][knight] & 0x1818000000ULL)
			bv[i] += 1;
		if (pieces[i][knight] & 0x3C3C3C3C0000ULL)
			bv[i] += 1;
		//bishops on principle diagonals
		if (pieces[i][bishop] & (0x8040201008040201ULL | 0x0804020180402010ULL))
			bv[i] += 1;
		if (pieces[i][bishop] & (0xC0E070381C0E0703ULL | 0x03070E1C3870E0C0ULL))
			bv[i] += 1;
		//rook on open file
		if (pieces[i][rook] & filea && !(pieces[i][pawn] & filea))
			bv[i] += 1;
		if (pieces[i][rook] & fileb && !(pieces[i][pawn] & fileb))
			bv[i] += 1;
		if (pieces[i][rook] & filec && !(pieces[i][pawn] & filec))
			bv[i] += 1;
		if (pieces[i][rook] & filed && !(pieces[i][pawn] & filed))
			bv[i] += 1;
		if (pieces[i][rook] & filee && !(pieces[i][pawn] & filee))
			bv[i] += 1;
		if (pieces[i][rook] & filef && !(pieces[i][pawn] & filef))
			bv[i] += 1;
		if (pieces[i][rook] & fileg && !(pieces[i][pawn] & fileg))
			bv[i] += 1;
		if (pieces[i][rook] & fileh && !(pieces[i][pawn] & fileh))
			bv[i] += 1;
	}
	return bv[movenotof] - bv[moveof];
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
	byte p = move->piece;

	moveno++;
	//8. if king moved first time mark move no.
	// check if king was moved. cannot castle then
	if (move->piece == king && !kingmoved[moveof])
		kingmoved[moveof] = moveno;
	//9. if rook moved first time mark move no.
	// check if rook was moved. in that case, cannot castle with that rook
	if (move->piece == rook && f & filea && !rookmoved[moveof][0])
		rookmoved[moveof][0] = moveno;
	else if (move->piece == rook && f & fileh && !rookmoved[moveof][1])
		rookmoved[moveof][1] = moveno;

	//1. move all
	all[moveof] ^= mov;
	//1.5. if promoted then remove pawn and add new piece
	//2. move individual piece
	if (pt) {
		pieces[moveof][pawn] ^= f;
		if (p != pawn)
			cout << "error@domove";
		pieces[moveof][pt] |= t;
	} else
		pieces[moveof][p] ^= mov;

	//3. do second move all
	//4. do second move individual piece
	if (mov2) {
		all[moveof] ^= mov2;
		pieces[moveof][rook] ^= mov2;
	}
	//7. if enpassent remove all & pawn
	//check which piece was that and move it
	//if this is enpassent then we need to adjust
	//for the capture of pawn at the right place
	if (move->flags & FLAGENPASSANT) {
		if (moveof == white) {
			all[movenotof] ^= t >> 8;
			pieces[movenotof][pawn] ^= t >> 8;
		} else {
			all[movenotof] ^= t << 8;
			pieces[movenotof][pawn] ^= t << 8;
		}
	} else if (move->captured) {
		//5. remove all
		all[movenotof] ^= t;
		//6. remove individual piece
		pieces[movenotof][move->captured] ^= t;
	}
	if (move->flags & FLAGDOUBLEMOVE) {
		if (moveof == white)
			epsq = t >> 8;
		else
			epsq = t << 8;
	} else
		epsq = 0;

	bitboard allpos = moveof == white ? all[white] : all[black];
	//record the move
	moveshistory.push(new moveshist(move, allpos));

	//change the sides
	moveof = movenotof;
	movenotof = (moveof == white ? black : white);
}

//input that we got from the user
cmove *Engine::input_move(char *m) {
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
		return NULL;

	//convert into int
	int from = ((m[1] - '1')) * 8;
	from += m[0] - 'a';
	int to = ((m[3] - '1')) * 8;
	to += m[2] - 'a';

	//convert into bitboards
	f <<= from;
	t <<= to;

	//if promotion then 'promoted to' required.
	//m[4] can be q, r, b, n for obvious reasons
	if ((f & pieces[white][pawn] && moveof == white && t & rank8) || (f
			& pieces[black][pawn] && moveof == black && t & rank1)) {
		if (strlen(m) < 5)
			return NULL;
		if (m[4] == 'q')
			pt = queen;
		else if (m[4] == 'r')
			pt = rook;
		else if (m[4] == 'b')
			pt = bishop;
		else if (m[4] == 'n')
			pt = knight;
		else
			return NULL;
	}

	cmove *mov = check_move(f, t, pt);
	if (mov)
		domove(mov);
	return mov;
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
	cmove *m = new cmove(f, t, mov2, moved, promto, flags, cap);

	//if in check reject the move
	domove(m);
	gen_atk_moves(moveof, atkmoves);
	if (pieces[movenotof][king] & atkmoves)
		incheck = 1;
	//check castling
	if (flags & FLAGCASTLEA) {
		if (moveof == black && 0x18 & atkmoves)
			incheck = 1;
		else if (moveof == white && 0x1800000000000000ULL & atkmoves)
			incheck = 1;
	} else if (flags & FLAGCASTLEH) {
		if (moveof == black && 0x30 & atkmoves)
			incheck = 1;
		else if (moveof == white && 0x3000000000000000ULL & atkmoves)
			incheck = 1;
	}
	undomove(m);
	if (incheck)
		return NULL;
	return m;
}

//generate moves for a given pawn position
void Engine::gen_pawn_moves(MoveNode *par) {
	// squares not occupied by any piece
	bitboard emptysq = ~(all[white] | all[black]);
	//iterate thru' all the pawns
	bitboard ap = pieces[moveof][pawn];//all pawns
	bitboard lsb;//last significant bit
	bitboard m, c;//moved to position, captured to position

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
				if (m & rank8)//pawn is promoted to...
				{
					par->addChild(create_move(lsb, m, pawn, queen));//queen
					par->addChild(create_move(lsb, m, pawn, rook));//rook
					par->addChild(create_move(lsb, m, pawn, bishop));//bishop
					par->addChild(create_move(lsb, m, pawn, knight));//knight
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
						par->addChild(create_move(lsb, m, pawn, 0,
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
				par->addChild(create_move(lsb, c, pawn, 0, FLAGENPASSANT));
			//make sure it captures adversary only
			c &= all[black];
			//if capture exists then push it
			if (c) {
				if (c & rank8)//pawn is promoted to...
				{
					par->addChild(create_move(lsb, c, pawn, queen));//queen
					par->addChild(create_move(lsb, c, pawn, rook));//rook
					par->addChild(create_move(lsb, c, pawn, bishop));//bishop
					par->addChild(create_move(lsb, c, pawn, knight));//knight
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
				if (c & rank8)//pawn is promoted to...
				{
					par->addChild(create_move(lsb, c, pawn, queen));//queen
					par->addChild(create_move(lsb, c, pawn, rook));//rook
					par->addChild(create_move(lsb, c, pawn, bishop));//bishop
					par->addChild(create_move(lsb, c, pawn, knight));//knight
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
				if (m & rank1)//pawn is promoted to...
				{
					par->addChild(create_move(lsb, m, pawn, queen));//queen
					par->addChild(create_move(lsb, m, pawn, rook));//rook
					par->addChild(create_move(lsb, m, pawn, bishop));//bishop
					par->addChild(create_move(lsb, m, pawn, knight));//knight
				} else
					par->addChild(create_move(lsb, m, pawn));
				//double move! for black it will be from rank 7.
				if (lsb & rank7) {
					m >>= 8;
					if (m & emptysq)
						par->addChild(create_move(lsb, m, pawn, 0,
								FLAGDOUBLEMOVE));
				}
			}
			//capture moves for black
			c = lsb >> 9;
			c &= ~fileh;
			if (c & epsq)
				par->addChild(create_move(lsb, c, pawn, 0, FLAGENPASSANT));
			c &= all[white];
			if (c) {
				if (c & rank1)//pawn is promoted to...
				{
					par->addChild(create_move(lsb, c, pawn, queen));//queen
					par->addChild(create_move(lsb, c, pawn, rook));//rook
					par->addChild(create_move(lsb, c, pawn, bishop));//bishop
					par->addChild(create_move(lsb, c, pawn, knight));//knight
				} else
					par->addChild(create_move(lsb, c, pawn));
			}
			c = lsb >> 7;
			c &= ~filea;
			if (c & epsq)
				par->addChild(create_move(lsb, c, pawn, 0, FLAGENPASSANT));
			c &= all[white];
			if (c) {
				if (c & rank1)//pawn is promoted to...
				{
					par->addChild(create_move(lsb, c, pawn, queen));//queen
					par->addChild(create_move(lsb, c, pawn, rook));//rook
					par->addChild(create_move(lsb, c, pawn, bishop));//bishop
					par->addChild(create_move(lsb, c, pawn, knight));//knight
				} else
					par->addChild(create_move(lsb, c, pawn));
			}
		}
		//remove the lsb from pawns bits
		ap ^= lsb;
	}
}
//generate attack moves for pawn
void Engine::gen_pawn_atk(side moveof, bitboard& atkbrd) {
	//iterate thru' all the pawns
	bitboard ap = pieces[moveof][pawn];//all pawns
	bitboard lsb;//last significant bit
	bitboard c;//moved to position, captured to position

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
	bitboard m;//moved to position
	int lsbint;
	bitboard ant/*all move to positions*/;
	lsbint = get_bit_pos(lsb);
	// get all moves on that position
	ant = king_moves[lsbint];

	// generate castling move if allowed
	if (!kingmoved[moveof]) {
		// castle towards filea
		if (!rookmoved[moveof][0]) {
			if (moveof == white && !(_all & 0xE) && (pieces[white][rook] & 0x1)) {
				par->addChild(create_move(lsb, 0x4, king, 0, FLAGCASTLEA
						| KINGMOVED, 0x1 | 0x8));
			} else if (moveof == black && !(_all & 0xE00000000000000ULL)
					&& pieces[black][rook] & 0x100000000000000ULL) {
				par->addChild(create_move(lsb, 0x400000000000000ULL, king, 0,
						FLAGCASTLEA | KINGMOVED, 0x100000000000000ULL
								| 0x800000000000000ULL));
			}
		}
		// castle towards fileh
		if (!rookmoved[moveof][1]) {
			if (moveof == white && !(_all & 0x60) && pieces[white][rook] & 0x80) {
				par->addChild(create_move(lsb, 0x40, king, 0, FLAGCASTLEH
						| KINGMOVED, 0x80 | 0x20));
			} else if (moveof == black && !(_all & 0x6000000000000000ULL)
					&& pieces[black][rook] & 0x8000000000000000ULL) {
				par->addChild(create_move(lsb, 0x4000000000000000ULL, king, 0,
						FLAGCASTLEH | KINGMOVED, 0x8000000000000000ULL
								| 0x2000000000000000ULL));
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
// generate king attack moves
void Engine::gen_king_atk(side moveof, bitboard& atkbrd) {
	//get king position
	bitboard lsb = pieces[moveof][king];
	int lsbint = get_bit_pos(lsb);
	// get all moves on that position
	atkbrd |= king_moves[lsbint] & all[movenotof];
}
// generate knight moves
void Engine::gen_knight_moves(MoveNode *par) {
	// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
	//get position
	bitboard an = pieces[moveof][knight];
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
				par->addChild(create_move(lsb, m, knight));
			}
			ant ^= m;
		}
		//remove the lsb from pawns bits
		an ^= lsb;
	}
}
// generate knight attack moves
void Engine::gen_knight_atk(side moveof, bitboard& atkbrd) {
	//get position
	bitboard an = pieces[moveof][knight];
	bitboard lsb;//last significant bit
	int lsbint;
	bitboard ant/*all move to positions*/;
	//iterate thru' all knights
	while (an) {
		lsb = an & (~an + 1);
		lsbint = get_bit_pos(lsb);
		// get all knight moves on that position
		ant = knight_moves[lsbint];
		// loop thru' all moves
		atkbrd |= ant & all[movenotof];
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
		_um = _um << 8 | _um << 16 | _um << 24 | _um << 32 | _um << 40 | _um
				<< 48;
		_um &= up_moves[lsbint];
		_um ^= up_moves[lsbint];
		_um &= othersq;

		// generate moves to the bottom
		_dm = down_moves[lsbint] & _all;
		_dm = _dm >> 8 | _dm >> 16 | _dm >> 24 | _dm >> 32 | _dm >> 40 | _dm
				>> 48;
		_dm &= down_moves[lsbint];
		_dm ^= down_moves[lsbint];
		_dm &= othersq;

		// loop thru' all moves
		_dm = _dm | _um | _lm | _rm;
		while (_dm) {
			m = _dm & (~_dm + 1);
			if (m & othersq) {
				if (piecefor == rook) {
					if (lsb & filea && rookmoved[moveof][0] == 0)
						flag = ROOKMOVED;
					else if (lsb & fileh && rookmoved[moveof][1] == 0)
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
void Engine::gen_rook_atk(side moveof, bitboard& atkbrd) {
	// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
	// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

	bitboard lsb;//last significant bit
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
		_um = _um << 8 | _um << 16 | _um << 24 | _um << 32 | _um << 40 | _um
				<< 48;
		_um &= up_moves[lsbint];
		_um ^= up_moves[lsbint];
		_um &= othersq;

		// generate moves to the bottom
		_dm = down_moves[lsbint] & _all;
		_dm = _dm >> 8 | _dm >> 16 | _dm >> 24 | _dm >> 32 | _dm >> 40 | _dm
				>> 48;
		_dm &= down_moves[lsbint];
		_dm ^= down_moves[lsbint];
		_dm &= othersq;

		// loop thru' all moves
		_dm = _dm | _um | _lm | _rm;
		atkbrd |= _dm & all[movenotof];
		ar ^= lsb;
	}
}
// generate bishop moves
void Engine::gen_bishop_moves(byte piecefor, bitboard ab, MoveNode *par) {
	// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
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
		_45m = _45m << 9 | _45m << 18 | _45m << 27 | _45m << 36 | _45m << 45
				| _45m << 54;
		_45m &= deg45_moves[lsbint];
		_45m ^= deg45_moves[lsbint];
		_45m &= othersq;

		// generate moves for left down
		_225m = deg225_moves[lsbint] & _all;
		_225m = _225m >> 9 | _225m >> 18 | _225m >> 27 | _225m >> 36 | _225m
				>> 45 | _225m >> 54;
		_225m &= deg225_moves[lsbint];
		_225m ^= deg225_moves[lsbint];
		_225m &= othersq;

		// generate moves right down
		_135m = deg135_moves[lsbint] & _all;
		_135m = _135m >> 7 | _135m >> 14 | _135m >> 21 | _135m >> 28 | _135m
				>> 35 | _135m >> 42;
		_135m &= deg135_moves[lsbint];
		_135m ^= deg135_moves[lsbint];
		_135m &= othersq;

		// generate moves for left up
		_315m = deg315_moves[lsbint] & _all;
		_315m = _315m << 7 | _315m << 14 | _315m << 21 | _315m << 28 | _315m
				<< 35 | _315m << 42;
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
	_45m = _45m << 9 | _45m << 18 | _45m << 27 | _45m << 36 | _45m << 45 | _45m
			<< 54;
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
void Engine::gen_bishop_atk(side moveof, bitboard& atkbrd) {
	// squares not occupied by our pieces
	bitboard othersq = ~all[moveof];
	// squares occupied by all pieces
	bitboard _all = all[white] | all[black];

	bitboard lsb;//last significant bit
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
		_225m = _225m >> 9 | _225m >> 18 | _225m >> 27 | _225m >> 36 | _225m
				>> 45 | _225m >> 54;
		_225m &= deg225_moves[lsbint];
		_225m ^= deg225_moves[lsbint];
		_225m &= othersq;

		// generate moves right down
		_135m = deg135_moves[lsbint] & _all;
		_135m = _135m >> 7 | _135m >> 14 | _135m >> 21 | _135m >> 28 | _135m
				>> 35 | _135m >> 42;
		_135m &= deg135_moves[lsbint];
		_135m ^= deg135_moves[lsbint];
		_135m &= othersq;

		// generate moves for left up
		_315m = deg315_moves[lsbint] & _all;
		_315m = _315m << 7 | _315m << 14 | _315m << 21 | _315m << 28 | _315m
				<< 35 | _315m << 42;
		_315m &= deg315_moves[lsbint];
		_315m ^= deg315_moves[lsbint];
		_315m &= othersq;

		// loop thru' all moves
		_315m = _315m | _135m | _225m | _45m;
		atkbrd |= _315m & all[movenotof];
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
//generate attack moves
void Engine::gen_atk_moves(side moveof, bitboard& atkbrd) {
	gen_king_atk(moveof, atkbrd);
	gen_pawn_atk(moveof, atkbrd);
	gen_knight_atk(moveof, atkbrd);
	gen_rook_atk(moveof, atkbrd);
	gen_bishop_atk(moveof, atkbrd);
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

	//if from is not valid return 0
	if (!(f & all[moveof]))
		return NULL;
	//if to is not valid return 0
	if (!(t & ~all[moveof]))
		return NULL;
	//get captured piece
	for (int i = 1; i < 6; i++) {
		if (t & pieces[movenotof][i]) {
			cap = i;
			break;
		}
	}
	//if piece move not valid return 0
	m = check_piece_move(f, t, promto, cap);
	if (!m)
		return NULL;

	domove(m);
	//if in check undo move and return 0
	gen_atk_moves(moveof, atkmoves);
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
				return new cmove(f, t, 0, pawn, promto);
			else if (t & (f << 16) & emptysq & rank4)
				return new cmove(f, t, 0, pawn, 0, FLAGDOUBLEMOVE);
		}
		//capture
		m = f << 9;
		m &= ~filea;
		if (t & m & epsq)
			return new cmove(f, t, 0, pawn, 0, FLAGENPASSANT);
		if (t & m & all[black])
			return new cmove(f, t, 0, pawn, promto, 0, cap);
		m = f << 7;
		m &= ~fileh;
		if (t & m & epsq)
			return new cmove(f, t, 0, pawn, 0, FLAGENPASSANT);
		if (t & m & all[black])
			return new cmove(f, t, 0, pawn, promto, 0, cap);

		//for black
		if ((f >> 8) & emptysq) {
			if (t & (f >> 8))
				return new cmove(f, t, 0, pawn, promto);
			else if (t & (f >> 16) & emptysq & rank5)
				return new cmove(f, t, 0, pawn, 0, FLAGDOUBLEMOVE);
		}
		//capture
		m = f >> 9;
		m &= ~fileh;
		if (t & m & epsq)
			return new cmove(f, t, 0, pawn, 0, FLAGENPASSANT);
		if (t & m & all[white])
			return new cmove(f, t, 0, pawn, promto, 0, cap);
		m = f >> 7;
		m &= ~filea;
		if (t & m & epsq)
			return new cmove(f, t, 0, pawn, 0, FLAGENPASSANT);
		if (t & m & all[white])
			return new cmove(f, t, 0, pawn, promto, 0, cap);
	} else if (f & pieces[moveof][queen]) {
		if (t & gen_rook_moves2(f, moveof) || t & gen_bishop_moves2(f, moveof))
			return new cmove(f, t, 0, queen, 0, 0, cap);
	} else if (f & pieces[moveof][rook]) {
		if (t & gen_rook_moves2(f, moveof))
			return new cmove(f, t, 0, rook, 0, ROOKMOVED, cap);
	} else if (f & pieces[moveof][bishop]) {
		if (t & gen_bishop_moves2(f, moveof))
			return new cmove(f, t, 0, bishop, 0, 0, cap);
	} else if (f & pieces[moveof][knight]) {
		int intf = get_bit_pos(f);
		m = knight_moves[intf] & ~all[moveof];
		if (t & m)
			return new cmove(f, t, 0, knight, 0, 0, cap);
	} else if (f & pieces[moveof][king]) {
		int intf = get_bit_pos(f);
		m = king_moves[intf] & ~all[moveof];
		if (t & m)
			return new cmove(f, t, 0, king, 0, KINGMOVED, cap);

		// generate castling move if allowed
		// need to check for check
		if (!kingmoved[moveof] && f & start_pieces[moveof][king]) { // if king hasnt moved
			// castle towards filea
			if (!rookmoved[moveof][0] && t & filec) { // if file A rook hasnt moved
				if (moveof == white && !((all[white] | all[black]) & 0xE)
						&& (pieces[white][rook] & 0x1)) {
					return new cmove(f, t, 0x1 | 0x8, king, 0, FLAGCASTLEA
							| KINGMOVED);
				} else if (moveof == black && !((all[white] | all[black])
						& 0xE00000000000000ULL) && pieces[black][rook]
						& 0x100000000000000ULL) {
					return new cmove(f, t, 0x100000000000000ULL
							| 0x800000000000000ULL, king, 0, FLAGCASTLEA
							| KINGMOVED);
				}
			}
			// castle towards fileh
			if (!rookmoved[moveof][1] && t & fileg) {
				if (moveof == white && !((all[white] | all[black]) & 0x60)
						&& pieces[white][rook] & 0x80) {
					return new cmove(f, t, 0x80 | 0x20, king, 0, FLAGCASTLEH
							| KINGMOVED);
				} else if (moveof == black && !((all[white] | all[black])
						& 0x6000000000000000ULL) && pieces[black][rook]
						& 0x8000000000000000ULL) {
					return new cmove(f, t, 0x8000000000000000ULL
							| 0x2000000000000000ULL, king, 0, FLAGCASTLEH
							| KINGMOVED);
				}
			}
		}
	} else {
		cout << "error\n";
	}

	return NULL;
}

void Engine::list_moves() {
	MoveNode *dummy = new MoveNode;
	generate_moves(dummy);
	for (MoveNode *cur = dummy->child; cur; cur = cur->next)
		cout << cur->move->getMoveTxt() << "; ";
	cout << endl;
}
