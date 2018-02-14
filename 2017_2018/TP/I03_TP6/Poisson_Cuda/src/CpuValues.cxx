#include "CpuValues.hxx"
#include "f.hxx"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

CpuValues::CpuValues(const CpuParameters * prm) : AbstractValues(prm)
{
	m_u = NULL;
	CpuValues::allocate(nn);
}


void CpuValues::allocate(size_t n)
{
	deallocate();
	m_u = new double [n];
}

void CpuValues::deallocate()
{
	if (m_u == NULL) {
		delete [] m_u;
		m_u = NULL;
	}
}

void CpuValues::init()
{
	int i, j, k;
	int imin = m_p->imin(0);
	int jmin = m_p->imin(1);
	int kmin = m_p->imin(2);

	int imax = m_p->imax(0);
	int jmax = m_p->imax(1);
	int kmax = m_p->imax(2);

	for (i=imin; i<imax; i++)
		for (j=jmin; j<jmax; j++)
			for (k=kmin; k<kmax; k++)
				operator()(i,j,k) = 0.0;
}

void CpuValues::init_f()
{
	int i, j, k;
	int imin = m_p->imin(0);
	int jmin = m_p->imin(1);
	int kmin = m_p->imin(2);

	int imax = m_p->imax(0);
	int jmax = m_p->imax(1);
	int kmax = m_p->imax(2);

	double dx = m_p->dx(0), dy = m_p->dx(1), dz = m_p->dx(2);
	double xmin =  m_p->xmin(0);
	double ymin =  m_p->xmin(1);
	double zmin =  m_p->xmin(2);

	for (i=imin; i<imax; i++)
		for (j=jmin; j<jmax; j++)
			for (k=kmin; k<kmax; k++)
				operator()(i,j,k) = f_CPU(xmin + i*dx, ymin + j*dy, zmin + k*dz);
}

void CpuValues::operator= (const CpuValues &other)
{
	int i;
	size_t nn = 1;

	for (i=0; i<3; i++)
		nn *= (m_n[i] = other.m_n[i]);

	allocate(nn);
	memcpy(m_u, other.m_u, nn*sizeof(double));
}
