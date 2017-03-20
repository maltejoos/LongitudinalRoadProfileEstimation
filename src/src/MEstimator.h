/*
 * MEstimator.h
 *
 *  Created on: 19.07.2011
 *      Author: joos
 */

#include <opencv/cv.h>
#include "ObservationEquation.h"

#ifndef MESTIMATOR_H_
#define MESTIMATOR_H_

struct MEstimator
{
	MEstimator(int numberOfSplines, int maxIterations, float cauchyFactor, float errorBound);

	//weighted observation Equations after estimate() has finished
	cv::Mat H_weighted;
	cv::Mat z_weighted;

	cv::Mat c;
	cv::Mat errors;

	int numberOfSplines;
	int numberOfDBP;

	//Options
	int maxIterations;
	float errorBound;
	float cauchyFactor;

	// run M-estimator
	//		vDisparity_f:	(filtered and weighted) vDisparity histogram
	void estimate(cv::Mat& vDisparity_f); //Not SSE optimized yet
};

#endif /* MESTIMATOR_H_ */
