/*
 * CubicBSpline.h
 *
 *  Created on: 19.07.2011
 *      Author: joos
 */

#include <opencv/cv.h>

#ifndef CUBICBSPLINE_H_
#define CUBICBSPLINE_H_

struct CubicBSpline
{
	// return 4 base values of uniform cubic B-spline at "parameter"
	//		parameter	:	the parameter to evaluate the base function. parameter e [0, 1)
	// 		des_Values_d: 	the 4 base values at "parameter"
	static void CUBaseFunctions(double parameter, cv::Mat& dest_Values_d);

	// multiplicate the deBoor points to the base values at "parameter" e [0, 1)
	static float cubicBSplinePart(cv::Mat deBoorPoints_d, double parameter);

	// evaluate "value" for cubic B-spline described by the deBoor points
	//		deBoorPoints_d	:	deBoor points of spline
	//		value			:	abscissa at which cubic spline shall be evaluated e [0, maxValue)
	//		maxValue		:	maximum abscissa of B-spline
	//		return			:	B-spline value at abscissa "value"
	static float cubicBSpline(cv::Mat deBoorPoints_d, double value, double maxValue);

	// get B-spline sample
	//		deBoorPoints_d	:	deBoor points of spline
	//		stepSize		:	step size at which B-Spline is evaluated
	//		maxValue		:	maximum abscissa of B-spline
	//		dest_Sample_i	:	evaluated sample
	static void getSample(cv::Mat deBoorPoints_d, double stepSize, double maxValue, cv::Mat& dest_Sample_i);

	//	plot Sample using openCV
	static void plotSample(cv::Mat& img_f, cv::Mat& sample_i, const cv::Scalar& color);
};

#endif /* CUBICBSPLINE_H_ */
