/*
 * AbstractValues.hxx
 *
 *  Created on: 12 févr. 2018
 *      Author: marc
 */

#ifndef ABSTRACTVALUES_HXX_
#define ABSTRACTVALUES_HXX_

#include "AbstractParameters.hxx"

class AbstractValues {
public:

	AbstractValues(const AbstractParameters * prm);
	virtual ~AbstractValues() {}

	virtual void init() = 0;
	virtual void init_f() = 0;

	void operator=(const AbstractValues & other);
	void print(std::ostream & f) const;
	void swap(AbstractValues & other);
	void plot(int order) const;
    double * data() { return m_u; }
    const double * data() const { return m_u; }

    double & operator() (int i,int j,int k) {
      return m_u[n2*i + n1*j + k];
    }
    double operator() (int i,int j,int k) const {
      return m_u[n2*i + n1*j + k];
    }

    std::string codeName;
    std::string deviceName;

protected:
	virtual void allocate(size_t n) = 0;
    virtual	void deallocate() = 0;

    int n1, n2, nn;
    double * m_u;
    int m_n[3];
    const AbstractParameters * m_p;

};

#endif /* ABSTRACTVALUES_HXX_ */
