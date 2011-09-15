#include "board.h"
#include "MoveGenerator.h"
#include "Evaluator.h"
#include "ChessEngine.h"

#include <vector>

void generateAndListMoves(board &b) {
	MoveGenerator mg;
	vector<bitmove*> v;

	mg.generate(b, v);

	for (unsigned int i = 0; i < v.size(); i++) {
		cout << *v[i] << ", ";
	}
	cout << endl;
	v.clear();
}

int main2() {
	char res[50];
	board b;
	Evaluator e;
	ChessEngine ce;
	int xforce=1;

	int engineplay = 0;
	bitmove m;

	for (;;) {
		if (!xforce && ((b.moveof == white && engineplay == PLAYWHITE)
				|| (b.moveof == black && engineplay == PLAYBLACK))) {
			ce.think(&b, m);
			b.domove(m);
			cout << "move " << m << endl;
		}
		cin.getline(res, 500);
		if (!strcmp(res, "q") || !strcmp(res, "quit"))
			break;
		else if (!strcmp(res, "xboard")) {
			// started by WinBoard
		}else if (!strncmp(res, "protover", 8)) {
			cout << "feature myname=\""
					<< ENGINEFULLNAME<<"\" time=0 reuse=0 analyze=0 done=1\n";
			cout.flush();
		} else if (!strcmp(res, "random")) {
			// ignore random command
		} else if (!strncmp(res, "level", 5)) {
			//Set time controls. need to parse.
		} else if (!strcmp(res, "hard")) {
			//Turn on pondering (thinking on the opponent's time,
			//also known as "permanent brain").
		} else if (!strncmp(res, "time", 4)) {
			//Set a clock that always belongs to the engine.
		} else if (!strncmp(res, "otim", 4)) {
			//Set a clock that always belongs to the opponent.
		} else if (!strcmp(res, "post")) {
			//Turn on thinking/pondering output.
		} else if (!strcmp(res, "black")) {
			engineplay = PLAYBLACK;
            xforce = 1;
		} else if (!strcmp(res, "white")) {
			engineplay = PLAYWHITE;
            xforce = 1;
		} else if (!strcmp(res, "force")) {
			/* Set the engine to play neither color ("force mode").
			 * Stop clocks.
			 * The engine should check that moves received in force mode are
			 * legal and made in the proper turn, but should not think,
			 * ponder, or make moves of its own.
			 */
			engineplay = 0;
            xforce = 1;
		} else if (!strcmp(res, "go")) {
			/*
			 * Leave force mode and set the engine to play the color that is on move.
			 * Associate the engine's clock with the color that is on move,
			 * the opponent's clock with the color that is not on move.
			 * Start the engine's clock.
			 * Start thinking and eventually make a move.
			 */
			engineplay = (b.moveof == white ? PLAYWHITE : PLAYBLACK);
            xforce = 0;
		} else if (!strcmp(res, "new")) {
			/*
			 * Reset the board to the standard chess starting position.
			 * Set White on move.
			 * Leave force mode and set the engine to play Black.
			 */
			b.newgame();
            engineplay = PLAYBLACK;
            xforce = 0;
		} else if (!strncmp(res, "accepted", 8)) {
			// features accepted
		} else if (!strcmp(res, "u") || !strcmp(res, "undo"))
			b.undolastmove();
		else if (!strcmp(res, "l") || !strcmp(res, "list"))
			generateAndListMoves(b);
		else if (!strcmp(res, "e") || !strcmp(res, "evaluate"))
			cout << "Score: " << e.score(b) << endl;
		else if (!strcmp(res, "t") || !strcmp(res, "think")) {
			ce.think(&b, m);
			b.domove(m);
		} else if (m.set(res)) {
			b.domove(m);
		} else
			cout << "Illigal move: "<<res<<endl;;
	}
	return 1;
}
