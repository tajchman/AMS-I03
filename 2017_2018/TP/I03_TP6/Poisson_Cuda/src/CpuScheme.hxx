/*
 * scheme.hxx
 *
 *  Created on: 5 janv. 2016
 *      Author: tajchman
 */

#ifndef SCHEME_HXX_
#define SCHEME_HXX_

#include <vector>
#include "parameters.hxx"
#include "AbstractScheme.hxx"

class CpuScheme : public AbstractScheme {

public:
  CpuScheme(const Parameters *P);
  virtual ~CpuScheme();

  bool iteration();

protected:
};

#endif /* SCHEME_HXX_ */
