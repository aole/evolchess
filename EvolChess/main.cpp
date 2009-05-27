/*
 * main.cpp
 *
 *  Created on: May 1, 2009
 *      Author: Bhupendra Aole
 */

#include <iostream>

#include "engine.h"

using namespace std;

//main function; entry point
int main() {
     int xboard = 0, xforce = 1;
     byte engineplay = 0;
     Engine engine;
     cmove *move;

     cin.rdbuf()->pubsetbuf(NULL,0);
     srand ( time(NULL) );

 	 cout<<"\nWelcome to Bhupendra Aole's Evolution Chess Program!\n";
     cout<<"Version 0.05 Date Created: 1-Apr-2009\n";
	 cout<<"Copyright 2009 Bhupendra Aole\n\n";

     //response buffer
     char res[500];

     for (;;) {
    	 //check if engine has to move
    	 if (!engine.gameended && !xforce) {
			 if (((engine.sidetomove() == white) && (engineplay & PLAYWHITE)) ||
				((engine.sidetomove() == black) && (engineplay & PLAYBLACK))) {
				 move = engine.doaimove();
				 if (engine.isDraw()) {
					 cout << "1/2-1/2";
					 engine.gameended = 1;
				 }
				 if (!move) {
					 if (engine.sidetomove() == black)
						 cout << "1-0 {White mates}\n";
					 else
						 cout << "0-1 {Black mates}\n";
					 engine.gameended = 1;
				 }
				 else
					 cout << "move " << move->getMoveTxt() << "\n";
				 if (engine.isMateMove()){
					 if (engine.sidetomove() == black)
						 cout << "1-0 {White mates}\n";
					 else
						 cout << "0-1 {Black mates}\n";
					 engine.gameended = 1;
				 }
				 if (!xboard)
					engine.show_board();
			 }
    	 }

    	 //get user/xboard input
         cin.getline(res, 500);

         if (!strcmp(res, "xboard")) {
            // started by WinBoard
            xboard = 1;
         } else if (!strncmp(res, "protover", 8)) {
                cout << "feature myname=\"Evolution Chess\" time=0\n";
                cout.flush();
         } else if (!strcmp(res, "quit")) {
                //exit program
                break;
         } else if (!strncmp(res, "accepted", 8)) {
                // features accepted
         } else if (!strcmp(res, "new")) {
             /*
              * Reset the board to the standard chess starting position.
              * Set White on move.
              * Leave force mode and set the engine to play Black.
              */
        	 engine.newGame();
             if (!xboard)
            	 engine.show_board();
             engineplay = PLAYBLACK;
             xforce = 0;
         } else if (!strcmp(res, "random")) {
             // ignore random command
         } else if (!strncmp (res, "level", 5)) {
             //Set time controls. need to parse.
         } else if (!strcmp (res, "hard")) {
             //Turn on pondering (thinking on the opponent's time,
        	 //also known as "permanent brain").
         } else if (!strncmp (res, "time", 4)) {
             //Set a clock that always belongs to the engine.
         } else if (!strncmp (res, "otim", 4)) {
             //Set a clock that always belongs to the opponent.
         } else if (!strcmp (res, "post")) {
             //Turn on thinking/pondering output.
         } else if (!strcmp (res, "force")) {
             //Set the engine to play neither color ("force mode").
        	 engineplay = 0;
         } else if (!strncmp (res, "result", 6)) {
        	 // game ended
        	 engine.gameended = 1;
        	 if (!xboard)
        		 cout<<res;
         } else if (!strcmp(res, "?")) {
			cout<<">> Available Commands:\n";
			cout<<">> new	: Start a new game.\n";
			cout<<">> show	: Show the current state of the game.\n";
			cout<<">> ls    : Displays list of possible moves.\n";
			cout<<">> <move>: Enter move in format e2e4, b8c6 etc.\n";
			cout<<">> ?	: Ofcourse, this help.\n";
			cout<<">> exit	: Bye; see u; tata; chao.\n\n";
		} else if (!strcmp(res, "ls")) {
			engine.list_moves();
		} else if (!strcmp(res, "show")) {
			engine.show_board();
		} else if (!strcmp(res, "undo")) {
			engine.undolastmove();
            if (!xboard)
            	engine.show_board();
		} else if (!strcmp(res, "black")) {
            engineplay = PLAYBLACK;
            xforce = 1;
		} else if (!strcmp(res, "white")) {
            engineplay = PLAYWHITE;
            xforce = 1;
		} else if (!strcmp(res, "computer")) {
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
			engineplay = (engine.sidetomove()==white?PLAYWHITE:PLAYBLACK);
            xforce = 0;
		} else if (engine.input_move(res)) {
            //got user/xboard move
			if (!xboard)
				engine.show_board();
         } else {
            cout << "Illegal move: " << res << endl;
         }
     }

     cout << "Thank u for playing! Have a nice Day...\n";
     return 0;
}
