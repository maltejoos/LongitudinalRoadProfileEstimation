/*
 * ObservationEquation.h
 *
 *  Created on: 19.07.2011
 *      Author: joos
 */

#include <opencv/cv.h>

#ifndef OBSERVATIONEQUATION_H_
#define OBSERVATIONEQUATION_H_

struct ObservationEquation
{
	//H*x=z
	cv::Mat H;
	cv::Mat z;

	// create observer and measurement matrices from image points
	//	vDisparity_f	:	input vDisparity histogram
	//	numberOfSplines	:	number of spline parts estimated spline shall exist of
	ObservationEquation(cv::Mat& vDisparity_f, int numberOfSplines);

	private:
	int countValidPixels(cv::Mat& vDisparity_f);
};

#endif /* OBSERVATIONEQUATION_H_ */
