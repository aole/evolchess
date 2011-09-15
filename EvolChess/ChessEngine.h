/*
 * ChessEngine.h
 *
 *  Created on: Sep 12, 2011
 *      Author: Bhupendra Aole
 */

#ifndef CHESSENGINE_H_
#define CHESSENGINE_H_

#include "board.h"
#include "bitmove.h"
#include "Evaluator.h"
#include "MoveGenerator.h"

#include "vector"

#define MAX_AI_SEARCH_DEPTH 5

enum SearchResult {
	MOVES,
	MATED,
	MATINGMOVE,
	DRAW
};

class PVLine {
public:
	int num; // Number of moves in the line.
	bitmove argmove[10]; // The line.

	PVLine() {
		num = 0;
	}
	void print() {
		for (int i = 0; i < num; i++) {
			cout << argmove[i] << " ";
		}
	}
};

class ChessEngine {
private:
	Evaluator e;
	MoveGenerator mg;

	board *b;
public:
	ChessEngine() {
	}
	virtual ~ChessEngine() {
	}

	SearchResult think(board *brd, bitmove &m);
	int alphabeta(int ply, int depth, int alpha, int beta, PVLine &pline);
	int evaluate() {
		return e.score(*b);
	}
	int generate(vector<bitmove*> &v) {
		mg.generate(*b, v);
		return v.size();
	}
	void domove(bitmove *m) {
		b->domove(*m);
	}
	void undomove() {
		b->undolastmove();
	}
	int islegal() {
		bitboard atk;
		mg.generateaktmoves(b, atk);
		return (!(b->ppieces[b->movenotof][king]&atk));
	}
};

#endif /* CHESSENGINE_H_ */
