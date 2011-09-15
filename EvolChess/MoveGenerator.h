/*
 * MoveGenerator.h
 *
 *  Created on: Sep 11, 2011
 *      Author: baole
 */

#ifndef MOVEGENERATOR_H_
#define MOVEGENERATOR_H_

#include <vector>

#include "constants.h"
#include "board.h"

class MoveGenerator {
private:
	void gen_king_moves(board &b, vector<bitmove*> &v);
	void gen_queen_moves(board &b, vector<bitmove*> &v);
	void gen_rook_moves(board &b, vector<bitmove*> &v);
	void gen_bishop_moves(board &b, vector<bitmove*> &v);
	void gen_knight_moves(board &b, vector<bitmove*> &v);
	void gen_pawn_moves(board &b, vector<bitmove*> &v);

	void gen_king_atk(board *b, bitboard& atkbrd);
	void gen_pawn_atk(board *b, bitboard& atkbrd);
	void gen_knight_atk(board *b, bitboard& atkbrd);
	void gen_rook_atk(board *b, bitboard& atkbrd);
	void gen_bishop_atk(board *b, bitboard& atkbrd);

public:
	MoveGenerator();
	virtual ~MoveGenerator();

	int get_bit_pos(bitboard b);

	void generate (board &b, vector<bitmove*> &v);

	void generateaktmoves(board *b, bitboard &atk) {
		gen_king_atk(b, atk);
		gen_pawn_atk(b, atk);
		gen_knight_atk(b, atk);
		gen_rook_atk(b, atk);
		gen_bishop_atk(b, atk);
	}
};

#endif /* MOVEGENERATOR_H_ */
