/*
 * board.cpp
 *
 *  Created on: Sep 4, 2011
 *      Author: baole
 */

#include "board.h"

board::board(){
	isready = 0;
}
void board::newgame(){
	isready = 1;
}
void board::domove(bitmove m) {
	// adjust all pieces bitboard
	all[moveof] ^= m.move;
	//pieces[moveof]
	moveof = moveof==white?black:white;
}
void board::undmove(bitmove m) {
	moveof = moveof==white?black:white;
	all[moveof] ^= m.move;
}
void board::undolastmove() {
}
