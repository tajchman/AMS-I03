/*
 * AbstractValues.hxx
 *
 *  Created on: 12 févr. 2018
 *      Author: marc
 */

#ifndef ABSTRACTVALUES_HXX_
#define ABSTRACTVALUES_HXX_

#include "parameters.hxx"

class AbstractValues {
public:

	AbstractValues(const Parameters * prm);
	virtual ~AbstractValues();

	void init(double (*f)(double, double, double) = 0L);
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

    int n1, n2;
    double * m_u;
    int m_n[3];
    const Parameters * m_p;

};

#endif /* ABSTRACTVALUES_HXX_ */
