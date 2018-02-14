/*
 * scheme.hxx
 *
 *  Created on: 5 janv. 2016
 *      Author: tajchman
 */

#ifndef CPUSCHEME_HXX_
#define CPUSCHEME_HXX_

#include <vector>
#include "CpuParameters.hxx"
#include "AbstractScheme.hxx"

class CpuScheme : public AbstractScheme {

public:
  CpuScheme(const CpuParameters *P);
  virtual ~CpuScheme();

  bool iteration();

protected:
};

#endif /* CPUSCHEME_HXX_ */
