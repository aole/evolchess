/*
 * ChessEngine.cpp
 *
 *  Created on: Sep 12, 2011
 *      Author: Bhupendra Aole
 */

#include "ChessEngine.h"

#include <time.h>

SearchResult ChessEngine::think(board *brd, bitmove & m) {
	b = brd;
	PVLine line;
	time_t t1;
	//start timer
	t1 = clock();

	int depth = MAX_AI_SEARCH_DEPTH;
	//for (int depth = 1; depth <= MAX_AI_SEARCH_DEPTH; depth++) {
		int ply = 0;
		// return book move if available
		// run alpha-beta
		alphabeta(ply, depth, -VALUEINFINITE, VALUEINFINITE, line);
	//}
	// return move from PV line
	// and/or the result
	if (!line.argmove[0].from) {
		//no moves left to do
		// TODO: check for stalemate
		cout << "#got mated\n";
		return MATED;
	} else if (!line.argmove[1].from) {
		//no moves left after this one
		// TODO: check for stalemate
		cout << "#mating move\n";
		m.copy(line.argmove[0]);
		return MATINGMOVE;
	}
	m.copy(line.argmove[0]);

	time_t t2 = clock() - t1;
	cout << "time taken:" << t2 << endl;

	return MOVES;
}

int ChessEngine::alphabeta(int ply, int depth, int alpha, int beta,
		PVLine & pline) {
	int pvfound = 0;
	bitmove *m;
	// if at leaf node
	// return evaluation of current board position
	if (!depth) {
		pline.num = 0;
		return evaluate();
	}

	// generate moves
	vector<bitmove*> v;
	// if no moves found return low score
	if (!generate(v))
		return -piecevalue[king] * depth;

	// TODO: sort move order
	// search the best move (found in prev. iteration of iterative
	// deepening) first
	if (pline.num > ply) {
		for (unsigned int i=0;i<v.size();i++){
			if (pline.argmove[ply]==*v[i]){
				m = v[i];
				v.erase(v.begin()+i);
				v.push_back(m);
				break;
			}
		}
	}

	// loop thru' all moves
	int score;
	PVLine line;
	int betafound = 0;
	while (!v.empty()) {
		m = v.back();
		if (betafound) {
			v.pop_back();
			delete m;
			continue;
		}

		// play the move
		domove(m);
		if (!islegal()) {
			v.pop_back();
			undomove();
			delete m;
			continue;
		}
		// get score from alphabeta searching deeper
		if (pvfound) {
			score = -alphabeta(ply + 1, depth - 1, -alpha - 1, -alpha, line);
			if ((score > alpha) && (score < beta))
				score = -alphabeta(ply + 1, depth - 1, -beta, -alpha, line);
		} else
			score = -alphabeta(ply + 1, depth - 1, -beta, -alpha, line);
		// take the move back
		undomove();
		// if score > beta, return beta
		if (score > beta) {
			alpha = beta;
			betafound = 1;
		}
		// if score > alpha, save to alpha
		else if (score > alpha) {
			alpha = score;
			pline.argmove[0].copy(*m);
			memcpy(pline.argmove + 1, line.argmove, line.num * sizeof(bitmove));
			pline.num = line.num + 1;
			if (!ply) {
				cout << depth << " " << score << " 0 0 ";
				pline.print();
				cout << endl;
			}
			pvfound = 1;
		}
		v.pop_back();
		delete m;
	}
	// when searching is over, return alpha
	return alpha;
}

