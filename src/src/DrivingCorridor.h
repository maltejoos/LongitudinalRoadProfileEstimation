/*
 * DrivingCorridor.h
 *
 *  Created on: 11.08.2011
 *      Author: joos
 */

#include <opencv/cv.h>

#ifndef DRIVINGCORRIDOR_H_
#define DRIVINGCORRIDOR_H_

struct DrivingCorridor
{
	//shift ROI rectangle proportional to steeringAngle. Size of ROI is (imgsrc.rows, imgsrc.cols/2).
	static void filter(cv::Mat& imgsrc_f, cv::Mat& dest_f, double steeringAngle);
};


#endif /* DRIVINGCORRIDOR_H_ */
