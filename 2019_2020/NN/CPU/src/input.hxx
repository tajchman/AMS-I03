/*
 * input.hxx
 *
 *  Created on: 7 mars 2019
 *      Author: marc
 */

#ifndef INPUT_HXX_
#define INPUT_HXX_

#include "vector.hxx"

class labelStream;
class imageStream;

class input {
public:
	input(const char * labelName, const char * imagesName);
	~input();
	bool next(vector &p, double &v);
	size_t n() { return m_n; }

private	:
	labelStream *labelFile;
	imageStream *imageFile;
	std::vector<unsigned char> buffer;
	unsigned char vbuffer;
	size_t m_n;
};

#endif /* INPUT_HXX_ */
