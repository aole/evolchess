/* * * * * * * * * *
 * EvolChess
 * * * * * * * * * *
 * constants.h
 *
 *  Created on: May 1, 2009
 *      Author: Bhupendra Aole
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <stdint.h>

/*	64 bit variable stores 1 bit for each square.
	Starting from a8 to h1
*/
typedef uint64_t bitboard;
typedef unsigned char byte;

/*
Position of pieces. 1 indicates presence while 0 indicates absence
                    { Most Significant bit }
E.g.:	0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0
		1 1 1 1 1 1 1 1
		1 1 1 1 1 1 1 1  shows all white pieces on the board at the starting position
{ Least Significant bit }
*/
//0: white
//1: black
enum side { white, black };

// which side engine have to play
const byte PLAYWHITE = 1;
const byte PLAYBLACK = 2;

//Numeric representation of each piece
//0: king
//1: queen
//2: rook
//3: bishop
//4: knight
//5: pawn
enum piece { king, queen, rook, bishop, knight, pawn };

const char notationw[6][2] = { "K", "Q", "R", "B", "N", "P" };
const char notationb[6][2] = { "k", "q", "r", "b", "n", "p" };
const int  piecevalue[6] = {1000, 9, 5, 3, 3, 1};

//initial position of all white and black pieces
const bitboard start_all[2] = { 0xffff, 0xffff000000000000LLU };
//initial position of individual white and black pieces
const bitboard start_pieces[2][6] = {
	{0x0000000000000010LLU, 0x0000000000000008LLU, 0x81, 0x24, 0x42, 0xff00},
	{0x1000000000000000LLU, 0x0800000000000000LLU, 0x8100000000000000LLU, 0x2400000000000000LLU, 0x4200000000000000LLU, 0x00ff000000000000LLU}};

//used for calculations later
const bitboard rank8 = 0xff00000000000000LLU;
const bitboard rank7 = 0xff000000000000LLU;
const bitboard rank6 = 0xff0000000000LLU;
const bitboard rank5 = 0xff00000000LLU;
const bitboard rank4 = 0xff000000LLU;
const bitboard rank3 = 0xff0000;
const bitboard rank2 = 0xff00LLU;
const bitboard rank1 = 0xff;
const bitboard rank[] = {rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8};

const bitboard filea = 0x0101010101010101LLU;
const bitboard fileb = 0x0202020202020202LLU;
const bitboard filec = 0x0404040404040404LLU;
const bitboard filed = 0x0808080808080808LLU;
const bitboard filee = 0x1010101010101010LLU;
const bitboard filef = 0x2020202020202020LLU;
const bitboard fileg = 0x4040404040404040LLU;
const bitboard fileh = 0x8080808080808080LLU;
const bitboard file[] = {filea,fileb,filec,filed,filee,filef,fileg,fileh};

//FLAGS used for camparison of flags in move class
const byte FLAGDOUBLEMOVE = 1;
const byte FLAGENPASSANT = 2;
const byte FLAGCASTLEA = 4;
const byte FLAGCASTLEH = 8;
const byte KINGMOVED = 16;
const byte ROOKMOVED = 32;

/*
for implementation of moves on using bitboard goto
http://www.mayothi.com/nagaskakichess6.html
*/
// king moves
const bitboard king_moves [] = {
      0x302, 0x705, 0xE0A, 0x1D14,
      0x3828, 0x7050, 0xE0A0, 0xC040,
      0x30203LLU, 0x70507LLU, 0xE0A0ELLU, 0x1C141CLLU,
      0x382838LLU, 0x705070LLU, 0xE0A0E0LLU, 0xC040C0LLU,
      0x3020300LLU, 0x7050700LLU, 0xE0A0E00LLU, 0x1C141C00LLU,
      0x38283800LLU, 0x70507000LLU, 0xE0A0E000LLU, 0xC040C000LLU,
      0x302030000LLU, 0x705070000LLU, 0xE0A0E0000LLU, 0x1C141C0000LLU,
      0x3828380000LLU, 0x7050700000LLU, 0xE0A0E00000LLU, 0xC040C00000LLU,
      0x30203000000LLU, 0x70507000000LLU, 0xE0A0E000000LLU, 0x1C141C000000LLU,
      0x382838000000LLU, 0x705070000000LLU, 0xE0A0E0000000LLU, 0xC040C0000000LLU,
      0x3020300000000LLU, 0x7050700000000LLU, 0xE0A0E00000000LLU, 0x1C141C00000000LLU,
      0x38283800000000LLU, 0x70507000000000LLU, 0xE0A0E000000000LLU, 0xC040C000000000LLU,
      0x302030000000000LLU, 0x705070000000000LLU, 0xE0A0E0000000000LLU, 0x1C141C0000000000LLU,
      0x3828380000000000LLU, 0x7050700000000000LLU, 0xE0A0E00000000000LLU, 0xC040C00000000000LLU,
      0x203000000000000LLU, 0x507000000000000LLU, 0xA0E000000000000LLU, 0x141C000000000000LLU,
      0x2838000000000000LLU, 0x5070000000000000LLU, 0xA0E0000000000000LLU, 0x40C0000000000000LLU };
// knight moves
const bitboard knight_moves [] = {
      0x20400, 0x50800, 0xA1100LLU , 0x142200LLU,
      0x284400LLU, 0x508800LLU, 0xA01000LLU, 0x402000LLU,
      0x2040004LLU, 0x5080008LLU, 0xA110011LLU, 0x14220022LLU,
      0x28440044LLU, 0x50880088LLU, 0xA0100010LLU, 0x40200020LLU,
      0x0204000402LLU, 0x0508000805LLU, 0x0A1100110ALLU, 0x1422002214LLU,
      0x2844004428LLU, 0x5088008850LLU, 0xA0100010A0LLU, 0x4020002040LLU,
      0x020400040200LLU, 0x050800080500LLU, 0x0A1100110A00LLU, 0x142200221400LLU,
      0x284400442800LLU, 0x508800885000LLU, 0xA0100010A000LLU, 0x402000204000LLU,
      0x02040004020000LLU, 0x05080008050000LLU, 0x0A1100110A0000LLU, 0x14220022140000LLU,
      0x28440044280000LLU, 0x50880088500000LLU, 0xA0100010A00000LLU, 0x40200020400000LLU,
      0x0204000402000000LLU, 0x0508000805000000LLU, 0x0A1100110A000000LLU, 0x1422002214000000LLU,
      0x2844004428000000LLU, 0x5088008850000000LLU, 0xA0100010A0000000LLU, 0x4020002040000000LLU,
      0x0400040200000000LLU, 0x0800080400000000LLU, 0x1100110A00000000LLU, 0x2200221400000000LLU,
      0x4400442800000000LLU, 0x8800885000000000LLU, 0x100010A000000000LLU, 0x2000204000000000LLU,
      0x04020000000000LLU, 0x08050000000000LLU, 0x110A0000000000LLU, 0x22140000000000LLU,
      0x44280000000000LLU, 0x88500000000000LLU, 0x10A00000000000LLU, 0x20400000000000LLU };
// all the squres on right of the position
// used for rook and queen
const bitboard right_moves [] = {
      0xFELLU, 0xFCLLU, 0xF8LLU, 0xF0LLU,
      0xE0LLU, 0xC0LLU, 0x80LLU, 0,
      0xFE00LLU, 0xFC00LLU, 0xF800LLU, 0xF000LLU,
      0xE000LLU, 0xC000LLU, 0x8000LLU, 0,
      0xFE0000LLU, 0xFC0000LLU, 0xF80000LLU, 0xF00000LLU,
      0xE00000LLU, 0xC00000LLU, 0x800000LLU, 0,
      0xFE000000LLU, 0xFC000000LLU, 0xF8000000LLU, 0xF0000000LLU,
      0xE0000000LLU, 0xC0000000LLU, 0x80000000LLU, 0,
      0xFE00000000LLU, 0xFC00000000LLU, 0xF800000000LLU, 0xF000000000LLU,
      0xE000000000LLU, 0xC000000000LLU, 0x8000000000LLU, 0,
      0xFE0000000000LLU, 0xFC0000000000LLU, 0xF80000000000LLU, 0xF00000000000LLU,
      0xE00000000000LLU, 0xC00000000000LLU, 0x800000000000LLU, 0,
      0xFE000000000000LLU, 0xFC000000000000LLU, 0xF8000000000000LLU, 0xF0000000000000LLU,
      0xE0000000000000LLU, 0xC0000000000000LLU, 0x80000000000000LLU, 0,
      0xFE00000000000000LLU, 0xFC00000000000000LLU, 0xF800000000000000LLU, 0xF000000000000000LLU,
      0xE000000000000000LLU, 0xC000000000000000LLU, 0x8000000000000000LLU, 0 };
// all the squres on left of the position
// used for rook and queen
const bitboard left_moves [] = {
      0, 0x1, 0x3, 0x7,
      0xF, 0x1F, 0x3F, 0x7F,
      0, 0x100LLU, 0x300LLU, 0x700LLU,
      0xF00LLU, 0x1F00LLU, 0x3F00LLU, 0x7F00LLU,
      0, 0x10000LLU, 0x30000LLU, 0x70000LLU,
      0xF0000LLU, 0x1F0000LLU, 0x3F0000LLU, 0x7F0000LLU,
      0, 0x1000000LLU, 0x3000000LLU, 0x7000000LLU,
      0xF000000LLU, 0x1F000000LLU, 0x3F000000LLU, 0x7F000000LLU,
      0, 0x100000000LLU, 0x300000000LLU, 0x700000000LLU,
      0xF00000000LLU, 0x1F00000000LLU, 0x3F00000000LLU, 0x7F00000000LLU,
      0, 0x10000000000LLU, 0x30000000000LLU, 0x70000000000LLU,
      0xF0000000000LLU, 0x1F0000000000LLU, 0x3F0000000000LLU, 0x7F0000000000LLU,
      0, 0x1000000000000LLU, 0x3000000000000LLU, 0x7000000000000LLU,
      0xF000000000000LLU, 0x1F000000000000LLU, 0x3F000000000000LLU, 0x7F000000000000LLU,
      0, 0x100000000000000LLU, 0x300000000000000LLU, 0x700000000000000LLU,
      0xF00000000000000LLU, 0x1F00000000000000LLU, 0x3F00000000000000LLU, 0x7F00000000000000LLU};
// all the squres up of the position
// used for rook and queen
const bitboard up_moves [] = {
      0x0101010101010100LLU, 0x0202020202020200LLU, 0x0404040404040400LLU, 0x0808080808080800LLU,
      0x1010101010101000LLU, 0x2020202020202000LLU, 0x4040404040404000LLU, 0x8080808080808000LLU,
      0x0101010101010000LLU, 0x0202020202020000LLU, 0x0404040404040000LLU, 0x0808080808080000LLU,
      0x1010101010100000LLU, 0x2020202020200000LLU, 0x4040404040400000LLU, 0x8080808080800000LLU,
      0x0101010101000000LLU, 0x0202020202000000LLU, 0x0404040404000000LLU, 0x0808080808000000LLU,
      0x1010101010000000LLU, 0x2020202020000000LLU, 0x4040404040000000LLU, 0x8080808080000000LLU,
      0x0101010100000000LLU, 0x0202020200000000LLU, 0x0404040400000000LLU, 0x0808080800000000LLU,
      0x1010101000000000LLU, 0x2020202000000000LLU, 0x4040404000000000LLU, 0x8080808000000000LLU,
      0x0101010000000000LLU, 0x0202020000000000LLU, 0x0404040000000000LLU, 0x0808080000000000LLU,
      0x1010100000000000LLU, 0x2020200000000000LLU, 0x4040400000000000LLU, 0x8080800000000000LLU,
      0x0101000000000000LLU, 0x0202000000000000LLU, 0x0404000000000000LLU, 0x0808000000000000LLU,
      0x1010000000000000LLU, 0x2020000000000000LLU, 0x4040000000000000LLU, 0x8080000000000000LLU,
      0x0100000000000000LLU, 0x0200000000000000LLU, 0x0400000000000000LLU, 0x0800000000000000LLU,
      0x1000000000000000LLU, 0x2000000000000000LLU, 0x4000000000000000LLU, 0x8000000000000000LLU,
      0,0,0,0,
      0,0,0,0 };
// all the squres down of the position
// used for rook and queen
const bitboard down_moves [] = {
      0,0,0,0,
      0,0,0,0,
      0x01LLU, 0x02LLU, 0x04LLU, 0x08LLU,
      0x10LLU, 0x20LLU, 0x40LLU, 0x80LLU,
      0x0101LLU, 0x0202LLU, 0x0404LLU, 0x0808LLU,
      0x1010LLU, 0x2020LLU, 0x4040LLU, 0x8080LLU,
      0x010101LLU, 0x020202LLU, 0x040404LLU, 0x080808LLU,
      0x101010LLU, 0x202020LLU, 0x404040LLU, 0x808080LLU,
      0x01010101LLU, 0x02020202LLU, 0x04040404LLU, 0x08080808LLU,
      0x10101010LLU, 0x20202020LLU, 0x40404040LLU, 0x80808080LLU,
      0x0101010101LLU, 0x0202020202LLU, 0x0404040404LLU, 0x0808080808LLU,
      0x1010101010LLU, 0x2020202020LLU, 0x4040404040LLU, 0x8080808080LLU,
      0x010101010101LLU, 0x020202020202LLU, 0x040404040404LLU, 0x080808080808LLU,
      0x101010101010LLU, 0x202020202020LLU, 0x404040404040LLU, 0x808080808080LLU,
      0x01010101010101LLU, 0x02020202020202LLU, 0x04040404040404LLU, 0x08080808080808LLU,
      0x10101010101010LLU, 0x20202020202020LLU, 0x40404040404040LLU, 0x80808080808080LLU};
// all the square 45 deg right (from north) of the position
// used for bishop and queen
const bitboard deg45_moves [] = {
      0x8040201008040200LLU, 0x80402010080400LLU, 0x804020100800LLU, 0x8040201000LLU,
      0x80402000LLU, 0x804000LLU, 0x8000LLU, 0,
      0x4020100804020000LLU, 0x8040201008040000LLU, 0x80402010080000LLU, 0x804020100000LLU,
      0x8040200000LLU, 0x80400000LLU, 0x800000LLU, 0,
      0x2010080402000000LLU, 0x4020100804000000LLU, 0x8040201008000000LLU, 0x80402010000000LLU,
      0x804020000000LLU, 0x8040000000LLU, 0x80000000LLU, 0,
      0x1008040200000000LLU, 0x2010080400000000LLU, 0x4020100800000000LLU, 0x8040201000000000LLU,
      0x80402000000000LLU, 0x804000000000LLU, 0x8000000000LLU, 0,
      0x0804020000000000LLU, 0x1008040000000000LLU, 0x2010080000000000LLU, 0x4020100000000000LLU,
      0x8040200000000000LLU, 0x80400000000000LLU, 0x800000000000LLU, 0,
      0x0402000000000000LLU, 0x0804000000000000LLU, 0x1008000000000000LLU, 0x2010000000000000LLU,
      0x4020000000000000LLU, 0x8040000000000000LLU, 0x80000000000000LLU, 0,
      0x0200000000000000LLU, 0x0400000000000000LLU, 0x0800000000000000LLU, 0x1000000000000000LLU,
      0x2000000000000000LLU, 0x4000000000000000LLU, 0x8000000000000000LLU, 0,
      0,0,0,0,0,0,0,0 };
// all the square 225 deg right (from north) of the position
// used for bishop and queen
const bitboard deg225_moves [] = {
      0,0,0,0,
      0,0,0,0,//1
      0, 0x1, 0x2, 0x4,
      0x8, 0x10, 0x20, 0x40,//2
      0, 0x100, 0x201, 0x402,
      0x804LLU, 0x1008LLU, 0x2010LLU, 0x4020LLU,//3
      0, 0x10000LLU, 0x20100LLU, 0x40201LLU,
      0x80402LLU, 0x100804LLU, 0x201008LLU, 0x402010LLU,//4
      0, 0x1000000LLU, 0x2010000LLU, 0x4020100LLU,
      0x8040201LLU, 0x10080402LLU, 0x20100804LLU, 0x40201008LLU,//5
      0, 0x100000000LLU, 0x201000000LLU, 0x402010000LLU,
      0x804020100LLU, 0x1008040201LLU, 0x2010080402LLU, 0x4020100804LLU,//6
      0, 0x10000000000LLU, 0x20100000000LLU, 0x40201000000LLU,
      0x80402010000LLU, 0x100804020100LLU, 0x201008040201LLU, 0x402010080402LLU,//7
      0, 0x1000000000000LLU, 0x2010000000000LLU, 0x4020100000000LLU,
      0x8040201000000LLU, 0x10080402010000LLU, 0x20100804020100LLU, 0x40201008040201LLU,//8
      };
// all the square 135 deg right (from north) of the position
// used for bishop and queen
const bitboard deg135_moves [] = {
      0,0,0,0,
      0,0,0,0,//1
      0x2LLU, 0x4LLU, 0x8LLU, 0x10LLU,
      0x20LLU, 0x40LLU, 0x80LLU, 0,//2
      0x204LLU, 0x408LLU, 0x810LLU, 0x1020LLU,
      0x2040LLU, 0x4080LLU, 0x8000LLU, 0,//3
      0x20408LLU, 0x40810LLU, 0x81020LLU, 0x102040LLU,
      0x204080LLU, 0x408000LLU, 0x800000LLU, 0,//4
      0x2040810LLU, 0x4081020LLU, 0x8102040LLU, 0x10204080LLU,
      0x20408000LLU, 0x40800000LLU, 0x80000000LLU, 0,//5
      0x204081020LLU, 0x408102040LLU, 0x810204080LLU, 0x1020408000LLU,
      0x2040800000LLU, 0x4080000000LLU, 0x8000000000LLU, 0,//6
      0x20408102040LLU, 0x40810204080LLU, 0x81020408000LLU, 0x102040800000LLU,
      0x204080000000LLU, 0x408000000000LLU, 0x800000000000LLU, 0,//7
      0x2040810204008LLU, 0x4081020408000LLU, 0x8102040800000LLU, 0x10204080000000LLU,
      0x20408000000000LLU, 0x40800000000000LLU, 0x80000000000000LLU, 0//8
      };
// all the square 315 deg right (from north) of the position
// used for bishop and queen
const bitboard deg315_moves [] = {
      0, 0x100LLU, 0x10200LLU, 0x1020400LLU,
      0x102040800LLU, 0x10204081000LLU, 0x1020408102000LLU, 0x102040810204000LLU,//1
      0, 0x10000LLU, 0x1020000LLU, 0x102040000LLU,
      0x10204080000LLU, 0x1020408100000LLU, 0x102040810200000LLU, 0x204081020400000LLU,//2
      0, 0x1000000LLU, 0x102000000LLU, 0x10204000000LLU,
      0x1020408000000LLU, 0x102040810000000LLU, 0x204081020000000LLU, 0x408102040000000LLU,//3
      0, 0x100000000LLU, 0x10200000000LLU, 0x1020400000000LLU,
      0x102040800000000LLU, 0x204081000000000LLU, 0x408102000000000LLU, 0x810204000000000LLU,//4
      0, 0x10000000000LLU, 0x1020000000000LLU, 0x102040000000000LLU,
      0x204080000000000LLU, 0x408100000000000LLU, 0x810200000000000LLU, 0x1020400000000000LLU,//5
      0, 0x1000000000000LLU, 0x102000000000000LLU, 0x204000000000000LLU,
      0x408000000000000LLU, 0x810000000000000LLU, 0x1020000000000000LLU, 0x2040000000000000LLU,//6
      0, 0x100000000000000LLU, 0x200000000000000LLU, 0x400000000000000LLU,
      0x800000000000000LLU, 0x1000000000000000LLU, 0x2000000000000000LLU, 0x4000000000000000LLU,//7
      0,0,0,0,
      0,0,0,0,//8
      };

#endif /* CONSTANTS_H_ */
