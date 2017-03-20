/*
 * CubicBSpline.cpp
 *
 *  Created on: 19.07.2011
 *      Author: joos
 */

#include "CubicBSpline.h"

#define ZERO_CMP 0.00001

void CubicBSpline::CUBaseFunctions(double t, cv::Mat& ret)
{
	//return 4 base values of uniform cubic B spline at parameter t

	static cv::Mat basis = (cv::Mat_<double>(4, 4) << -1./6., 3./6., -3./6., 1./6., 3./6., -6./6., 3./6., 0.0, -3./6., 0.0, 3./6., 0.0, 1./6., 4./6., 1./6., 0.0);
	cv::Mat t_vec = (cv::Mat_<double>(1, 4) << t*t*t, t*t, t, 1);

	ret = t_vec*basis;
}


float CubicBSpline::cubicBSplinePart(cv::Mat c, double t)
{
	// S=Base(t)*deBoorPoints

	cv::Mat res(1, 4, CV_64FC1);
	CUBaseFunctions(t, res);
	cv::Mat ret(1, 1, CV_64FC1);
	ret = res*c;
	return ret.at<double>(0,0);
}

float CubicBSpline::cubicBSpline(cv::Mat c, double d, double maxValue)
{
	int numberOfSplines = c.rows-3;
	double knotDistance = (double)maxValue/(double)numberOfSplines;

	//determine related spline part
	int currSplinePart = floor(d/knotDistance);

	//store deBoor points of related spline part
	cv::Mat curr_c = (cv::Mat_<double>(4, 1) << c.at<double>(currSplinePart, 0), c.at<double>(currSplinePart+1, 0), c.at<double>(currSplinePart+2, 0), c.at<double>(currSplinePart+3, 0));

	//map disparity to curve parameter t e[0,1)
	double t = (d-currSplinePart*knotDistance)/knotDistance;

	//return spline value
	return cubicBSplinePart(curr_c, t);
}

void CubicBSpline::getSample(cv::Mat c, double stepSize, double maxValue, cv::Mat& sample)
{
	int numberOfSplines = c.rows-3;
	int numberOfDBP = c.rows;
	double knotDistance = (double)maxValue/(double)numberOfSplines;

	sample.create(((double)numberOfSplines)/stepSize, 2, CV_32SC1);

	int counter=0;
	for(int i = 0; i<numberOfDBP-3; ++i)
	{
		cv::Mat curr_c = (cv::Mat_<double>(4, 1) << c.at<double>(i, 0), c.at<double>(i+1, 0), c.at<double>(i+2, 0), c.at<double>(i+3, 0));

		for(double t=0; t<1-ZERO_CMP; t+=stepSize)
		{
			sample.at<int>(counter, 0) = t*knotDistance + i*knotDistance; //remap curve parameter t e[0,1)
			sample.at<int>(counter, 1) = cubicBSplinePart(curr_c, t);

			++counter;
		}
	}
}

void CubicBSpline::plotSample(cv::Mat& img, cv::Mat& sample, const cv::Scalar& color)
{
	cv::Point p;
	for(int i=0; i<sample.rows-10; ++i)
	{
		p.x = sample.at<int>(i, 0);
		p.y = sample.at<int>(i, 1);
		cv::circle(img, p, 1, color);
	}
}
