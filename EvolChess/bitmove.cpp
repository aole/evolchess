/*
 * bitmove.cpp
 *
 *  Created on: Sep 4, 2011
 *      Author: baole
 */

#include "bitmove.h"

int bitmove::set(const char *m) {
	from = 1;
	to = 1;
	promto = none;
	//check if string is legal
	/*m[0] and m[2] can be a, b, c, d, e, f, g, h
	 m[1] and m[3] can be 1, 2, 3, 4, 5, 6, 7, 8
	 */
	if (m[0] < 'a' || m[0] > 'h' || m[1] < '1' || m[1] > '8' || m[2] < 'a'
			|| m[2] > 'h' || m[3] < '1' || m[3] > '8')
		return 0;

	//convert into int
	int f = ((m[1] - '1')) * 8;
	f += m[0] - 'a';
	int t = ((m[3] - '1')) * 8;
	t += m[2] - 'a';

	//convert into bitboards
	from <<= f;
	to <<= t;

	if (strlen(m) == 5) {
		if (m[4] == 'q')
			promto = queen;
		else if (m[4] == 'r')
			promto = rook;
		else if (m[4] == 'b')
			promto = bishop;
		else if (m[4] == 'n')
			promto = knight;
		else
			return 0;
	}

	return 1;
}

ostream &operator<<(ostream &s, bitmove m) {
	if (m.from & filea)
		s << 'a';
	else if (m.from & fileb)
		s << 'b';
	else if (m.from & filec)
		s << 'c';
	else if (m.from & filed)
		s << 'd';
	else if (m.from & filee)
		s << 'e';
	else if (m.from & filef)
		s << 'f';
	else if (m.from & fileg)
		s << 'g';
	else if (m.from & fileh)
		s << 'h';

	if (m.from & rank1)
		s << '1';
	else if (m.from & rank2)
		s << '2';
	else if (m.from & rank3)
		s << '3';
	else if (m.from & rank4)
		s << '4';
	else if (m.from & rank5)
		s << '5';
	else if (m.from & rank6)
		s << '6';
	else if (m.from & rank7)
		s << '7';
	else if (m.from & rank8)
		s << '8';

	if (m.to & filea)
		s << 'a';
	else if (m.to & fileb)
		s << 'b';
	else if (m.to & filec)
		s << 'c';
	else if (m.to & filed)
		s << 'd';
	else if (m.to & filee)
		s << 'e';
	else if (m.to & filef)
		s << 'f';
	else if (m.to & fileg)
		s << 'g';
	else if (m.to & fileh)
		s << 'h';

	if (m.to & rank1)
		s << '1';
	else if (m.to & rank2)
		s << '2';
	else if (m.to & rank3)
		s << '3';
	else if (m.to & rank4)
		s << '4';
	else if (m.to & rank5)
		s << '5';
	else if (m.to & rank6)
		s << '6';
	else if (m.to & rank7)
		s << '7';
	else if (m.to & rank8)
		s << '8';

	if (m.promto < 5 && m.promto > 0)
		s << notationb[m.promto][0];

	return s;
}
