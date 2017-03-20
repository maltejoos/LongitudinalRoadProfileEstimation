/*
 * EasyVisualOdometry.cpp
 *
 *  Created on: 20.07.2011
 *      Author: joos
 */

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "EasyVisualOdometry.h"

EasyVisualOdometry::EasyVisualOdometry(float f_, float cu_, float cv_, float b_) : matcher(3, 50, 50, 200, 1, 5, 5, 1, 1, 1)//default values
{
	visualOdometry.setCalibration(f_, cu_, cv_, b_);
}

void EasyVisualOdometry::pushImagePair(string leftImage, string rightImage)
{
	//read Images
	cv::Mat li, ri;
	li = cv::imread(leftImage, 0); //CV_8UC1
	ri = cv::imread(rightImage, 0); //CV_8UC1

	matcher.pushBack(li.data, ri.data, li.cols, li.rows, li.cols, false);
}


void EasyVisualOdometry::computeStep()
{
	matcher.matchFeatures(2);
	matcher.bucketFeatures( 4, 50, 50 ) ;

	if(!visualOdometry.update( matcher.getMatches(), 0.1, true, false )){
		cout << "Error in libviso2" << endl;
	}
}

void EasyVisualOdometry::getTransformation(cv::Mat& T)
{
	visoTrans = visualOdometry.getTransformation();

	T = (cv::Mat_<float>(4, 4) << 	visoTrans.val[0][0], visoTrans.val[0][1], visoTrans.val[0][2], visoTrans.val[0][3],
									visoTrans.val[1][0], visoTrans.val[1][1], visoTrans.val[1][2], visoTrans.val[1][3],
									visoTrans.val[2][0], visoTrans.val[2][1], visoTrans.val[2][2], visoTrans.val[2][3],
									visoTrans.val[3][0], visoTrans.val[3][1], visoTrans.val[3][2], visoTrans.val[3][3]);
}
