/*
 * ValidSampleRange.cpp
 *
 *  Created on: 21.07.2011
 *      Author: joos
 */

#include "ValidSampleRange.h"

ValidSampleRange::ValidSampleRange(int windowsize_, int sumThreshold_, int pcounterThreshold_)
{
	windowsize = windowsize_;
	sumThreshold = sumThreshold_;
	pcounterThreshold = pcounterThreshold_;

	minIndex=0;
	minValue=0;
	maxIndex=0;
	maxValue=0;
}

void ValidSampleRange::findSampleRange(cv::Mat& img, cv::Mat& sample)
{
	findMinLimit(img, sample);
	findMaxLimit(img, sample);
}

void ValidSampleRange::findMinLimit(cv::Mat& img, cv::Mat& sample)
{
	int row=0;

	//from center of sample to left end
	for(int i=sample.rows/2; i>=0; --i)
	{
		//get sample point
		int disp = sample.at<int>(i,0);
		row = sample.at<int>(i,1);

		//catch if sample point is too close to image border
		if(disp < windowsize || disp > img.cols-windowsize || row < windowsize || row > img.rows-windowsize){
			continue;
		}

		//define window
		cv::Mat tmpwindow = img(cv::Range(row-windowsize, row+windowsize), cv::Range(disp-windowsize, disp+windowsize));

		float sum = cv::sum(tmpwindow).val[0];
		int pcounter = cv::countNonZero(tmpwindow);

		//too few measurements and too less-weighted measurements
		if(sum <= sumThreshold && pcounter <= pcounterThreshold){
			minIndex = disp+windowsize;
			minValue = row;
			return;
		}
	}

	//sample is completely valid
	minIndex = 1; // 0 corresponds to distance infinity, so we use 1 as smallest disparity
	minValue = row;
}

void ValidSampleRange::findMaxLimit(cv::Mat& img, cv::Mat& sample)
{
	//find first non-empty pixel from lower right corner
	for(int row=img.rows-1; row>=0/*img.rows-20*/; --row)
	{
		for(int col=img.cols-1; col>=img.cols-50; --col)
		{
			//not empty
			if((int)img.at<float>(row, col))
			{
				maxIndex=col;
				maxValue=row;
				return;
			}
		}
	}

	//Default fallback
	maxIndex = img.cols;
	maxValue = img.rows-10;
}
