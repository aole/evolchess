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
     int xboard = 0;
     Engine engine;

     cin.rdbuf()->pubsetbuf(NULL,0);
     srand ( time(NULL) );

 	 cout<<"\nWelcome to Bhupendra Aole's Evolution Chess Program!\n";
     cout<<"Version 0.05 Date Created: 1-Apr-2009\n";
	 cout<<"Copyright 2009 Bhupendra Aole\n\n";

     //response buffer
     char res[500];

     for (;;) {
         //_read(0, res, 50);
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
                //cout << "starting new game ...\n";
        	 engine.newGame();
			 engine.generate_moves();
             if (!xboard)
            	 engine.show_board();
           // start new game
         } else if (!strcmp(res, "random")) {
                // ignore random command
         } else if (!strncmp (res, "level", 5)) {
                //Set time controls. need to parse.
         } else if (!strcmp (res, "hard")) {
                //Turn on pondering (thinking on the opponent's time, also known as "permanent brain").
         } else if (!strncmp (res, "time", 4)) {
                //Set a clock that always belongs to the engine.
         } else if (!strncmp (res, "otim", 4)) {
                //Set a clock that always belongs to the opponent.
         } else if (!strcmp (res, "post")) {
                //Turn on thinking/pondering output.
         } else if (!strcmp (res, "force")) {
                //Set the engine to play neither color ("force mode").
         } else if (!strcmp (res, "ls")) { //command line move
             //display list of moves
        	 engine.list_moves();
         } else if (!strcmp(res, "?")) {
			cout<<">> Available Commands:\n";
			cout<<">> new	: Start a new game.\n";
			cout<<">> show	: Show the current state of the game.\n";
			cout<<">> ls    : Displays list of possible moves.\n";
			cout<<">> <move>: Enter move in format e2e4, b8c6 etc.\n";
			cout<<">> ?	: Ofcourse, this help.\n";
			cout<<">> exit	: Bye; see u; tata; chao.\n\n";
		} else if (!strcmp(res, "show")) {
			engine.show_board();
		} else if (!strcmp(res, "undo")) {
			engine.undolastmove();
			engine.show_board();
		} else if (engine.domove(engine.input_move(res))>=0) {
                //cout << "got correct move\n";
                engine.generate_moves();
                if (!xboard)
                   engine.list_moves();
                cout << "move " << engine.getMoveTxt(engine.doaimove()) << "\n";
                engine.generate_moves();
                if (!xboard)
                   engine.show_board();
         } else {
                cout << "Error (unknown command): " << res << endl;
         }
     }
     cout << "Thank u for playing! Have a nice Day...\n";
     return 0;
}
