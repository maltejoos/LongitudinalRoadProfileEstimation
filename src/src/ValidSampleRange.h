/*
 * ValidSampleRange.h
 *
 *  Created on: 21.07.2011
 *      Author: joos
 */

#include <opencv/cv.h>

#ifndef VALIDSAMPLERANGE_H_
#define VALIDSAMPLERANGE_H_

class ValidSampleRange
{
	private:

	//values to determine where estimated sample is supported by enough measurements
		int windowsize; //half of window width
		float sumThreshold; //sum of pixels threshold
		int pcounterThreshold; //amount of pixels threshold

		void findMinLimit(cv::Mat& Image_f, cv::Mat& sample_i);
		void findMaxLimit(cv::Mat& Image_f, cv::Mat& sample_i);

	public:

		// windowsize			:	ROI(u,v) becomes: u-windowsize .. u+windowsize, v-windowsize .. v+windowsize
		// sumThreshold			:	threshold for accumulated sum of all pixels in ROI
		// pcounterThreshold	:	threshold for amount of non-zero pixels in ROI
		ValidSampleRange(int windowsize_, int sumThreshold_, int pcounterThreshold_);

		// valid borders of sample to be determined by findSampleRange()
		int minIndex;
		int minValue;
		int maxIndex;
		int maxValue;

		// find valid range of sample
		// sample is valid until sum(ROI) <= sumThreshold and amountOfPixels(ROI) <= pcounterThreshold
		// 	vDispImg_f	:	vDisparity histogram (filtered and weighted)
		//	sample_i	:	sample to be checked
		void findSampleRange(cv::Mat& vDispImg_f, cv::Mat& sample_i);
};

#endif /* VALIDSAMPLERANGE_H_ */
