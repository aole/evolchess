/*
 * Evaluator.h
 *
 *  Created on: Sep 11, 2011
 *      Author: Bhupendra Aole
 */

#ifndef EVALUATOR_H_
#define EVALUATOR_H_

#include "board.h"

class Evaluator {
public:
	Evaluator() {}
	virtual ~Evaluator() {}

	int score(board &b);
};

#endif /* EVALUATOR_H_ */
