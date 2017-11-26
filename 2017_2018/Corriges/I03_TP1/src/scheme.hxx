/*
 * scheme.hxx
 *
 *  Created on: 5 janv. 2016
 *      Author: tajchman
 */

#ifndef SCHEME_HXX_
#define SCHEME_HXX_

#include "values.hxx"
#include "parameters.hxx"

double iterate(const Values & u1, Values & u2,
               double dt, Parameters &P);
void exchange(Values &u, Parameters &P);

#endif /* SCHEME_HXX_ */
