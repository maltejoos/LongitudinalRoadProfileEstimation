/*
 * ElevationMap.h
 *
 *  Created on: 19.07.2011
 *      Author: joos
 */

#include <opencv/cv.h>
#include "RoadRepresentation.h"

#ifndef ELEVATIONMAP_H_
#define ELEVATIONMAP_H_

struct ElevationMap
{
	float baseWidth;		// [m]
	float focalLength;		// [px]
	float toleranceFactor; 	// factor multiplied by row of pixel to determine if pixel is roadpoint or not

	cv::Mat plainElevationMap;						//plain scalar elevation map
	cv::Mat_<cv::Vec3b> plainElevationMapColored_; 	//plain color encoded elevation map after executing draw(...)

	ElevationMap(float baseWidth, float focalLength, float toleranceFactor);

	// draws an elevation map on realImage relative to the road niveau
	//	dispImgOrg			:	original disparity image. Non-filtered
	//	dispImgUDFiltered_f	:	disparity image filtered by uDisparity (cf. DisparityHistogram::filterObstaclesFromUD(...))
	//	realImage_f			:	the image the elevation map will be overlaid
	//	roadRepresentation	:	road representation of current frame
	//	segmentRoad		:	if true, the road is segmented and colored blue. Else, the road will be colored according to height=0 (green)
	void draw(cv::Mat& dispImgOrg_f, cv::Mat& dispImgUDFiltered_f, cv::Mat& realImage_f, RoadRepresentation roadRepresentation, bool segmentRoad);
};

#endif /* ELEVATIONMAP_H_ */
