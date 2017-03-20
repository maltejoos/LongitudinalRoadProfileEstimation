/*
 * EasyVisualOdometry.h
 *
 *  Created on: 20.07.2011
 *      Author: joos
 */

#include <string>
#include "libviso2/src/visualodometry.h"
#include "libviso2/src/matcher.h"

using namespace std;

#ifndef EASYVISUALODOMETRY_H_
#define EASYVISUALODOMETRY_H_

class EasyVisualOdometry
{
	private:
		VisualOdometry visualOdometry;
		Matcher matcher;
		Matrix visoTrans;

	public:
		EasyVisualOdometry(float f, float cu, float cv, float b);

		void pushImagePair(string leftImage, string rightImage);
		void computeStep();
		void getTransformation(cv::Mat& transformationMatrix_f);
};

#endif /* EASYVISUALODOMETRY_H_ */
