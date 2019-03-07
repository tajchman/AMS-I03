
#include <vector>

#ifndef MATRIX_HXX_
#define MATRIX_HXX_

#include "vector.hxx"

class matrix {
public:
	matrix() : m_n(0), m_m(0) {}
	matrix(size_t n, size_t m) : m_v(n*m), m_n(n), m_m(m) {}

	void resize(size_t n, size_t m) {
		m_n = n;
		m_m = m;
		m_v.resize(m*n);
	}

	double operator()(size_t i, size_t j) const {
		return m_v[i * m_m + j];
	}

	double & operator()(size_t i, size_t j) {
		return m_v[i * m_m + j];
	}
    size_t n() const { return m_n; }
    size_t m() const { return m_m; }

private:
	size_t m_n, m_m;
	std::vector<double> m_v;
};

inline const vector operator*(const matrix &m, const vector & v) {
	vector w(m.n());
	double s;
	size_t i,j;
	for (i=0; i<m.n(); i++) {
		s = 0.0;
		for (j=0; j<m.m(); j++)
			s += m(i,j) * v[j];
		w[i] = s;
	}
	return w;
}

#endif /* MATRIX_HXX_ */
