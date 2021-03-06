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

#ifndef NULL
#ifdef __cplusplus
#define NULL 0
#else
#define NULL ((void*)0)
#endif
#endif

#define VERSION_MAJOR 0
#define VERSION_MINOR 6
#define BUILD 5

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
enum side { white = 0, black = 1 };

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

static const char notationw[6][2] = { "K", "Q", "R", "B", "N", "P" };
static const char notationb[6][2] = { "k", "q", "r", "b", "n", "p" };
static const int  piecevalue[6] = {100000, 900, 500, 300, 300, 100};
static const int  VALUEINFINITE = piecevalue[0];

//chess board constants
const bitboard a8 = 0x100000000000000ULL;
const bitboard b8 = 0x200000000000000ULL;
const bitboard c8 = 0x400000000000000ULL;
const bitboard d8 = 0x800000000000000ULL;
const bitboard e8 = 0x1000000000000000ULL;
const bitboard f8 = 0x2000000000000000ULL;
const bitboard g8 = 0x4000000000000000ULL;
const bitboard h8 = 0x8000000000000000ULL;

const bitboard a7 = 0x1000000000000ULL;
const bitboard b7 = 0x2000000000000ULL;
const bitboard c7 = 0x4000000000000ULL;
const bitboard d7 = 0x8000000000000ULL;
const bitboard e7 = 0x10000000000000ULL;
const bitboard f7 = 0x20000000000000ULL;
const bitboard g7 = 0x40000000000000ULL;
const bitboard h7 = 0x80000000000000ULL;

const bitboard a6 = 0x10000000000ULL;
const bitboard b6 = 0x20000000000ULL;
const bitboard c6 = 0x40000000000ULL;
const bitboard d6 = 0x80000000000ULL;
const bitboard e6 = 0x100000000000ULL;
const bitboard f6 = 0x200000000000ULL;
const bitboard g6 = 0x400000000000ULL;
const bitboard h6 = 0x800000000000ULL;

const bitboard a5 = 0x100000000ULL;
const bitboard b5 = 0x200000000ULL;
const bitboard c5 = 0x400000000ULL;
const bitboard d5 = 0x800000000ULL;
const bitboard e5 = 0x1000000000ULL;
const bitboard f5 = 0x2000000000ULL;
const bitboard g5 = 0x4000000000ULL;
const bitboard h5 = 0x8000000000ULL;

const bitboard a4 = 0x1000000ULL;
const bitboard b4 = 0x2000000ULL;
const bitboard c4 = 0x4000000ULL;
const bitboard d4 = 0x8000000ULL;
const bitboard e4 = 0x10000000ULL;
const bitboard f4 = 0x20000000ULL;
const bitboard g4 = 0x40000000ULL;
const bitboard h4 = 0x80000000ULL;

const bitboard a3 = 0x10000ULL;
const bitboard b3 = 0x20000ULL;
const bitboard c3 = 0x40000ULL;
const bitboard d3 = 0x80000ULL;
const bitboard e3 = 0x100000ULL;
const bitboard f3 = 0x200000ULL;
const bitboard g3 = 0x400000ULL;
const bitboard h3 = 0x800000ULL;

const bitboard a2 = 0x100ULL;
const bitboard b2 = 0x200ULL;
const bitboard c2 = 0x400ULL;
const bitboard d2 = 0x800ULL;
const bitboard e2 = 0x1000ULL;
const bitboard f2 = 0x2000ULL;
const bitboard g2 = 0x4000ULL;
const bitboard h2 = 0x8000ULL;

const bitboard a1 = 0x1ULL;
const bitboard b1 = 0x2ULL;
const bitboard c1 = 0x4ULL;
const bitboard d1 = 0x8ULL;
const bitboard e1 = 0x10ULL;
const bitboard f1 = 0x20ULL;
const bitboard g1 = 0x40ULL;
const bitboard h1 = 0x80ULL;

const bitboard rank8 = 0xff00000000000000ULL;
const bitboard rank7 = 0xff000000000000ULL;
const bitboard rank6 = 0xff0000000000ULL;
const bitboard rank5 = 0xff00000000ULL;
const bitboard rank4 = 0xff000000ULL;
const bitboard rank3 = 0xff0000ULL;
const bitboard rank2 = 0xff00ULL;
const bitboard rank1 = 0xffULL;
const bitboard rank[] = {rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8};

const bitboard filea = 0x0101010101010101ULL;
const bitboard fileb = 0x0202020202020202ULL;
const bitboard filec = 0x0404040404040404ULL;
const bitboard filed = 0x0808080808080808ULL;
const bitboard filee = 0x1010101010101010ULL;
const bitboard filef = 0x2020202020202020ULL;
const bitboard fileg = 0x4040404040404040ULL;
const bitboard fileh = 0x8080808080808080ULL;
const bitboard file[] = {filea,fileb,filec,filed,filee,filef,fileg,fileh};

//initial position of all white and black pieces
const bitboard start_all[2] = { rank1|rank2, rank7|rank8 };
//initial position of individual white and black pieces
const bitboard start_pieces[2][6] = {
	{e1, d1, a1|h1, c1|f1, b1|g1, rank2},
	{e8, d8, a8|h8, c8|f8, b8|g8, rank7}};

//FLAGS used for comparison of flags in move class
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
      0x302, 0x705, 0xE0A, 0x1C14,
      0x3828, 0x7050, 0xE0A0, 0xC040,
      0x30203ULL, 0x70507ULL, 0xE0A0EULL, 0x1C141CULL,
      0x382838ULL, 0x705070ULL, 0xE0A0E0ULL, 0xC040C0ULL,
      0x3020300ULL, 0x7050700ULL, 0xE0A0E00ULL, 0x1C141C00ULL,
      0x38283800ULL, 0x70507000ULL, 0xE0A0E000ULL, 0xC040C000ULL,
      0x302030000ULL, 0x705070000ULL, 0xE0A0E0000ULL, 0x1C141C0000ULL,
      0x3828380000ULL, 0x7050700000ULL, 0xE0A0E00000ULL, 0xC040C00000ULL,
      0x30203000000ULL, 0x70507000000ULL, 0xE0A0E000000ULL, 0x1C141C000000ULL,
      0x382838000000ULL, 0x705070000000ULL, 0xE0A0E0000000ULL, 0xC040C0000000ULL,
      0x3020300000000ULL, 0x7050700000000ULL, 0xE0A0E00000000ULL, 0x1C141C00000000ULL,
      0x38283800000000ULL, 0x70507000000000ULL, 0xE0A0E000000000ULL, 0xC040C000000000ULL,
      0x302030000000000ULL, 0x705070000000000ULL, 0xE0A0E0000000000ULL, 0x1C141C0000000000ULL,
      0x3828380000000000ULL, 0x7050700000000000ULL, 0xE0A0E00000000000ULL, 0xC040C00000000000ULL,
      0x203000000000000ULL, 0x507000000000000ULL, 0xA0E000000000000ULL, 0x141C000000000000ULL,
      0x2838000000000000ULL, 0x5070000000000000ULL, 0xA0E0000000000000ULL, 0x40C0000000000000ULL };
// knight moves
const bitboard knight_moves [] = {
      0x20400, 0x50800, 0xA1100ULL , 0x142200ULL,
      0x284400ULL, 0x508800ULL, 0xA01000ULL, 0x402000ULL,//1
      0x2040004ULL, 0x5080008ULL, 0xA110011ULL, 0x14220022ULL,
      0x28440044ULL, 0x50880088ULL, 0xA0100010ULL, 0x40200020ULL,//2
      0x0204000402ULL, 0x0508000805ULL, 0x0A1100110AULL, 0x1422002214ULL,
      0x2844004428ULL, 0x5088008850ULL, 0xA0100010A0ULL, 0x4020002040ULL,//3
      0x020400040200ULL, 0x050800080500ULL, 0x0A1100110A00ULL, 0x142200221400ULL,
      0x284400442800ULL, 0x508800885000ULL, 0xA0100010A000ULL, 0x402000204000ULL,//4
      0x02040004020000ULL, 0x05080008050000ULL, 0x0A1100110A0000ULL, 0x14220022140000ULL,
      0x28440044280000ULL, 0x50880088500000ULL, 0xA0100010A00000ULL, 0x40200020400000ULL,//5
      0x0204000402000000ULL, 0x0508000805000000ULL, 0x0A1100110A000000ULL, 0x1422002214000000ULL,
      0x2844004428000000ULL, 0x5088008850000000ULL, 0xA0100010A0000000ULL, 0x4020002040000000ULL,//6
      0x0400040200000000ULL, 0x0800080400000000ULL, 0x1100110A00000000ULL, 0x2200221400000000ULL,
      0x4400442800000000ULL, 0x8800885000000000ULL, 0x100010A000000000ULL, 0x2000204000000000ULL,//7
      0x04020000000000ULL, 0x08050000000000ULL, 0x110A0000000000ULL, 0x22140000000000ULL,
      0x44280000000000ULL, 0x88500000000000ULL, 0x10A00000000000ULL, 0x20400000000000ULL };//8
// all the squares on right of the position
// used for rook and queen
const bitboard right_moves [] = {
      0xFEULL, 0xFCULL, 0xF8ULL, 0xF0ULL,
      0xE0ULL, 0xC0ULL, 0x80ULL, 0,
      0xFE00ULL, 0xFC00ULL, 0xF800ULL, 0xF000ULL,
      0xE000ULL, 0xC000ULL, 0x8000ULL, 0,
      0xFE0000ULL, 0xFC0000ULL, 0xF80000ULL, 0xF00000ULL,
      0xE00000ULL, 0xC00000ULL, 0x800000ULL, 0,
      0xFE000000ULL, 0xFC000000ULL, 0xF8000000ULL, 0xF0000000ULL,
      0xE0000000ULL, 0xC0000000ULL, 0x80000000ULL, 0,
      0xFE00000000ULL, 0xFC00000000ULL, 0xF800000000ULL, 0xF000000000ULL,
      0xE000000000ULL, 0xC000000000ULL, 0x8000000000ULL, 0,
      0xFE0000000000ULL, 0xFC0000000000ULL, 0xF80000000000ULL, 0xF00000000000ULL,
      0xE00000000000ULL, 0xC00000000000ULL, 0x800000000000ULL, 0,
      0xFE000000000000ULL, 0xFC000000000000ULL, 0xF8000000000000ULL, 0xF0000000000000ULL,
      0xE0000000000000ULL, 0xC0000000000000ULL, 0x80000000000000ULL, 0,
      0xFE00000000000000ULL, 0xFC00000000000000ULL, 0xF800000000000000ULL, 0xF000000000000000ULL,
      0xE000000000000000ULL, 0xC000000000000000ULL, 0x8000000000000000ULL, 0 };
// all the squares on left of the position
// used for rook and queen
const bitboard left_moves [] = {
      0, 0x1, 0x3, 0x7,
      0xF, 0x1F, 0x3F, 0x7F,
      0, 0x100ULL, 0x300ULL, 0x700ULL,
      0xF00ULL, 0x1F00ULL, 0x3F00ULL, 0x7F00ULL,
      0, 0x10000ULL, 0x30000ULL, 0x70000ULL,
      0xF0000ULL, 0x1F0000ULL, 0x3F0000ULL, 0x7F0000ULL,
      0, 0x1000000ULL, 0x3000000ULL, 0x7000000ULL,
      0xF000000ULL, 0x1F000000ULL, 0x3F000000ULL, 0x7F000000ULL,
      0, 0x100000000ULL, 0x300000000ULL, 0x700000000ULL,
      0xF00000000ULL, 0x1F00000000ULL, 0x3F00000000ULL, 0x7F00000000ULL,
      0, 0x10000000000ULL, 0x30000000000ULL, 0x70000000000ULL,
      0xF0000000000ULL, 0x1F0000000000ULL, 0x3F0000000000ULL, 0x7F0000000000ULL,
      0, 0x1000000000000ULL, 0x3000000000000ULL, 0x7000000000000ULL,
      0xF000000000000ULL, 0x1F000000000000ULL, 0x3F000000000000ULL, 0x7F000000000000ULL,
      0, 0x100000000000000ULL, 0x300000000000000ULL, 0x700000000000000ULL,
      0xF00000000000000ULL, 0x1F00000000000000ULL, 0x3F00000000000000ULL, 0x7F00000000000000ULL};
// all the squares up of the position
// used for rook and queen
const bitboard up_moves [] = {
      0x0101010101010100ULL, 0x0202020202020200ULL, 0x0404040404040400ULL, 0x0808080808080800ULL,
      0x1010101010101000ULL, 0x2020202020202000ULL, 0x4040404040404000ULL, 0x8080808080808000ULL,
      0x0101010101010000ULL, 0x0202020202020000ULL, 0x0404040404040000ULL, 0x0808080808080000ULL,
      0x1010101010100000ULL, 0x2020202020200000ULL, 0x4040404040400000ULL, 0x8080808080800000ULL,
      0x0101010101000000ULL, 0x0202020202000000ULL, 0x0404040404000000ULL, 0x0808080808000000ULL,
      0x1010101010000000ULL, 0x2020202020000000ULL, 0x4040404040000000ULL, 0x8080808080000000ULL,
      0x0101010100000000ULL, 0x0202020200000000ULL, 0x0404040400000000ULL, 0x0808080800000000ULL,
      0x1010101000000000ULL, 0x2020202000000000ULL, 0x4040404000000000ULL, 0x8080808000000000ULL,
      0x0101010000000000ULL, 0x0202020000000000ULL, 0x0404040000000000ULL, 0x0808080000000000ULL,
      0x1010100000000000ULL, 0x2020200000000000ULL, 0x4040400000000000ULL, 0x8080800000000000ULL,
      0x0101000000000000ULL, 0x0202000000000000ULL, 0x0404000000000000ULL, 0x0808000000000000ULL,
      0x1010000000000000ULL, 0x2020000000000000ULL, 0x4040000000000000ULL, 0x8080000000000000ULL,
      0x0100000000000000ULL, 0x0200000000000000ULL, 0x0400000000000000ULL, 0x0800000000000000ULL,
      0x1000000000000000ULL, 0x2000000000000000ULL, 0x4000000000000000ULL, 0x8000000000000000ULL,
      0,0,0,0,
      0,0,0,0 };
// all the squares down of the position
// used for rook and queen
const bitboard down_moves [] = {
      0,0,0,0,
      0,0,0,0,
      0x01ULL, 0x02ULL, 0x04ULL, 0x08ULL,
      0x10ULL, 0x20ULL, 0x40ULL, 0x80ULL,
      0x0101ULL, 0x0202ULL, 0x0404ULL, 0x0808ULL,
      0x1010ULL, 0x2020ULL, 0x4040ULL, 0x8080ULL,
      0x010101ULL, 0x020202ULL, 0x040404ULL, 0x080808ULL,
      0x101010ULL, 0x202020ULL, 0x404040ULL, 0x808080ULL,
      0x01010101ULL, 0x02020202ULL, 0x04040404ULL, 0x08080808ULL,
      0x10101010ULL, 0x20202020ULL, 0x40404040ULL, 0x80808080ULL,
      0x0101010101ULL, 0x0202020202ULL, 0x0404040404ULL, 0x0808080808ULL,
      0x1010101010ULL, 0x2020202020ULL, 0x4040404040ULL, 0x8080808080ULL,
      0x010101010101ULL, 0x020202020202ULL, 0x040404040404ULL, 0x080808080808ULL,
      0x101010101010ULL, 0x202020202020ULL, 0x404040404040ULL, 0x808080808080ULL,
      0x01010101010101ULL, 0x02020202020202ULL, 0x04040404040404ULL, 0x08080808080808ULL,
      0x10101010101010ULL, 0x20202020202020ULL, 0x40404040404040ULL, 0x80808080808080ULL};
// all the square 45 deg right (from north) of the position
// used for bishop and queen
const bitboard deg45_moves [] = {
      0x8040201008040200ULL, 0x80402010080400ULL, 0x804020100800ULL, 0x8040201000ULL,
      0x80402000ULL, 0x804000ULL, 0x8000ULL, 0,//1
      0x4020100804020000ULL, 0x8040201008040000ULL, 0x80402010080000ULL, 0x804020100000ULL,
      0x8040200000ULL, 0x80400000ULL, 0x800000ULL, 0,//2
      0x2010080402000000ULL, 0x4020100804000000ULL, 0x8040201008000000ULL, 0x80402010000000ULL,
      0x804020000000ULL, 0x8040000000ULL, 0x80000000ULL, 0,//3
      0x1008040200000000ULL, 0x2010080400000000ULL, 0x4020100800000000ULL, 0x8040201000000000ULL,
      0x80402000000000ULL, 0x804000000000ULL, 0x8000000000ULL, 0,//4
      0x0804020000000000ULL, 0x1008040000000000ULL, 0x2010080000000000ULL, 0x4020100000000000ULL,
      0x8040200000000000ULL, 0x80400000000000ULL, 0x800000000000ULL, 0,//5
      0x0402000000000000ULL, 0x0804000000000000ULL, 0x1008000000000000ULL, 0x2010000000000000ULL,
      0x4020000000000000ULL, 0x8040000000000000ULL, 0x80000000000000ULL, 0,//6
      0x0200000000000000ULL, 0x0400000000000000ULL, 0x0800000000000000ULL, 0x1000000000000000ULL,
      0x2000000000000000ULL, 0x4000000000000000ULL, 0x8000000000000000ULL, 0,
      0,0,0,0,0,0,0,0 };
// all the square 225 deg right (from north) of the position
// used for bishop and queen
const bitboard deg225_moves [] = {
      0,0,0,0,
      0,0,0,0,//1
      0, 0x1, 0x2, 0x4,
      0x8, 0x10, 0x20, 0x40,//2
      0, 0x100, 0x201, 0x402,
      0x804ULL, 0x1008ULL, 0x2010ULL, 0x4020ULL,//3
      0, 0x10000ULL, 0x20100ULL, 0x40201ULL,
      0x80402ULL, 0x100804ULL, 0x201008ULL, 0x402010ULL,//4
      0, 0x1000000ULL, 0x2010000ULL, 0x4020100ULL,
      0x8040201ULL, 0x10080402ULL, 0x20100804ULL, 0x40201008ULL,//5
      0, 0x100000000ULL, 0x201000000ULL, 0x402010000ULL,
      0x804020100ULL, 0x1008040201ULL, 0x2010080402ULL, 0x4020100804ULL,//6
      0, 0x10000000000ULL, 0x20100000000ULL, 0x40201000000ULL,
      0x80402010000ULL, 0x100804020100ULL, 0x201008040201ULL, 0x402010080402ULL,//7
      0, 0x1000000000000ULL, 0x2010000000000ULL, 0x4020100000000ULL,
      0x8040201000000ULL, 0x10080402010000ULL, 0x20100804020100ULL, 0x40201008040201ULL,//8
      };
// all the square 135 deg right (from north) of the position
// used for bishop and queen
const bitboard deg135_moves [] = {
      0,0,0,0,
      0,0,0,0,//1
      0x2ULL, 0x4ULL, 0x8ULL, 0x10ULL,
      0x20ULL, 0x40ULL, 0x80ULL, 0,//2
      0x204ULL, 0x408ULL, 0x810ULL, 0x1020ULL,
      0x2040ULL, 0x4080ULL, 0x8000ULL, 0,//3
      0x20408ULL, 0x40810ULL, 0x81020ULL, 0x102040ULL,
      0x204080ULL, 0x408000ULL, 0x800000ULL, 0,//4
      0x2040810ULL, 0x4081020ULL, 0x8102040ULL, 0x10204080ULL,
      0x20408000ULL, 0x40800000ULL, 0x80000000ULL, 0,//5
      0x204081020ULL, 0x408102040ULL, 0x810204080ULL, 0x1020408000ULL,
      0x2040800000ULL, 0x4080000000ULL, 0x8000000000ULL, 0,//6
      0x20408102040ULL, 0x40810204080ULL, 0x81020408000ULL, 0x102040800000ULL,
      0x204080000000ULL, 0x408000000000ULL, 0x800000000000ULL, 0,//7
      0x2040810204008ULL, 0x4081020408000ULL, 0x8102040800000ULL, 0x10204080000000ULL,
      0x20408000000000ULL, 0x40800000000000ULL, 0x80000000000000ULL, 0//8
      };
// all the square 315 deg right (from north) of the position
// used for bishop and queen
const bitboard deg315_moves [] = {
      0, 0x100ULL, 0x10200ULL, 0x1020400ULL,
      0x102040800ULL, 0x10204081000ULL, 0x1020408102000ULL, 0x102040810204000ULL,//1
      0, 0x10000ULL, 0x1020000ULL, 0x102040000ULL,
      0x10204080000ULL, 0x1020408100000ULL, 0x102040810200000ULL, 0x204081020400000ULL,//2
      0, 0x1000000ULL, 0x102000000ULL, 0x10204000000ULL,
      0x1020408000000ULL, 0x102040810000000ULL, 0x204081020000000ULL, 0x408102040000000ULL,//3
      0, 0x100000000ULL, 0x10200000000ULL, 0x1020400000000ULL,
      0x102040800000000ULL, 0x204081000000000ULL, 0x408102000000000ULL, 0x810204000000000ULL,//4
      0, 0x10000000000ULL, 0x1020000000000ULL, 0x102040000000000ULL,
         0x204080000000000ULL, 0x408100000000000ULL, 0x810200000000000ULL, 0x1020400000000000ULL,//5
      0, 0x1000000000000ULL, 0x102000000000000ULL, 0x204000000000000ULL,
         0x408000000000000ULL, 0x810000000000000ULL, 0x1020000000000000ULL, 0x2040000000000000ULL,//6
      0, 0x100000000000000ULL, 0x200000000000000ULL,  0x400000000000000ULL,
         0x800000000000000ULL, 0x1000000000000000ULL, 0x2000000000000000ULL, 0x4000000000000000ULL,//7
      0,0,0,0,
      0,0,0,0,//8
      };

#endif /* CONSTANTS_H_ */
