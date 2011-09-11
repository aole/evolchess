#include "board.h"

int main2() {
	char res[50];
	board b;
	b.newgame();
	bitmove m;

	for (;;) {
		cin.getline(res, 500);
		if (!strcmp(res, "q"))
			break;
		if (!strcmp(res, "u"))
			b.undolastmove();
		else if (!m.set(res)) {
			cout << "illigal move";
		} else
			b.domove(m);
		cout << b;
	}
	return 1;
}
