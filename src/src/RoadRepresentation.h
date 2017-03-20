/*
 * RoadRepresentation.h
 *
 *  Created on: 19.07.2011
 *      Author: joos
 */

#include <opencv/cv.h>
#include "ValidSampleRange.h"

#ifndef ROADREPRESENTATION_H_
#define ROADREPRESENTATION_H_

struct RoadRepresentation
{
	ValidSampleRange validSampleRange;

	//cf. ValidSampleRange for parameter description
	RoadRepresentation(int windowsize_, int sumThreshold_, int pcounterThreshold_);

	// calculate lookup tables
	//		Image_f			:	(filtered and weighted) vDisparity Histogram
	//		deBoorPoints_d	:	deBoor points of estimated spline
	void calculateLUTs(cv::Mat& Image_f, cv::Mat& deBoorPoints_d);

	cv::Mat LUT_rowOfDisp;	//Lookup Table: roadrow(disparity). Type: int_32
	cv::Mat LUT_dispOfRow;	//Lookup Table: roaddisparity(imagerow). Type: int_32
};

#endif /* ROADREPRESENTATION_H_ */
