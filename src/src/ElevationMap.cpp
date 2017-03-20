/*
 * ElevationMap.cpp
 *
 *  Created on: 19.07.2011
 *      Author: joos
 */

#include "ElevationMap.h"

#define ZERO_CMP 0.00001

ElevationMap::ElevationMap(float baseWidth_, float focalLength_, float toleranceFactor_)
{
	baseWidth = baseWidth_;
	focalLength = focalLength_;
	toleranceFactor = toleranceFactor_;
}

void ElevationMap::draw(cv::Mat& dispImgOrg, cv::Mat& dispImgUDFiltered, cv::Mat& realImg, RoadRepresentation roadrep, bool segmentRoad)
{
	//"cast" to Mat_ class for channel access
	cv::Mat_<cv::Vec3b> realImg_ = realImg;

	plainElevationMapColored_ = cv::Mat::zeros(realImg_.size(), realImg_.type());
	plainElevationMap= cv::Mat::zeros(realImg_.size(), CV_32FC1);

	#pragma omp parallel for
	for(int row=0; row<roadrep.validSampleRange.maxValue; ++row)
	{
		for(int col=0; col<dispImgOrg.cols; ++col)
		{
			//disparity at dispImg(row, col)
			float currDisp = dispImgOrg.at<float>(row, col) + 0.5;

			//pixel empty
			if((int)currDisp==0){
				continue;
			}

			//invalid pixel because too far
			if(currDisp <= roadrep.validSampleRange.minIndex+1){
				continue;
			}

			bool tooClose=false;
			int tooCloseTolerance = 5;

			//point is closer than closest modeled road point (including tolerance). Do not interpret as road point.
			if(currDisp > roadrep.validSampleRange.maxIndex + tooCloseTolerance){
				tooClose=true;
			}

			//limit to max index of spline model (=> constant extrapolation)
			if(currDisp > roadrep.validSampleRange.maxIndex){
				currDisp = roadrep.validSampleRange.maxIndex;
			}

			//road row for current disparity
			int currRowOfDisp = roadrep.LUT_rowOfDisp.at<int>(currDisp, 0);

			//pixel filtered by uDisparity
			bool udf = false;
			if(currDisp > 0 && (int)dispImgUDFiltered.at<float>(row, col) == 0){
				udf = true;
			}

			//road point
			if(segmentRoad && !udf && !tooClose && row > currRowOfDisp-toleranceFactor*row && row < currRowOfDisp+toleranceFactor*row)
			{
				plainElevationMapColored_(row, col)[0] = 100;
				plainElevationMap.at<float>(row, col) = 0;
			}

			//below road niveau
			else if(segmentRoad && !tooClose && row > currRowOfDisp-toleranceFactor*row)
			{
				continue;

				//color cyan
				//plainElevationMapColored_(row, col)[0] = 255;
				//plainElevationMapColored_(row, col)[1] = 255;
				//plainElevationMapColored_(row, col)[2] = 255*0.06;
			}

			//elevation map
			else
			{
				//Calculate height
				//limit dispIndex to maxModelDisp
				int dispIndex = currDisp+0.5;
				if(dispIndex >= roadrep.validSampleRange.maxIndex){
					dispIndex = roadrep.validSampleRange.maxIndex-1;
				}
				int currRowOfDisp = roadrep.LUT_rowOfDisp.at<int>(dispIndex, 0);

				//intersection angle
				float alpha = acos((row*currRowOfDisp + focalLength*focalLength) / (sqrt(row*row+focalLength*focalLength) * sqrt(currRowOfDisp*currRowOfDisp + focalLength*focalLength)));

				//distances
				float r1 = baseWidth*focalLength/currDisp;
				float r2 = r1; //approximation

				//height
				float height = sqrt(r1*r1 + r2*r2 - 2*r1*r2*cos(alpha));

				plainElevationMap.at<float>(row, col) = height;

				//map height to HSV hue
				//height: 	0   => maxHeight
				//hue:		120 => 0
				float maxHeight = 1.5;

				//HSV values
				float H = 120*(1-height/maxHeight) ;
				H = H < 0 ? 0 : H;
				float S = 1;
				float V = 1;

				//RGB values
				float R=0, G=0, B=0;

				//Convert HSV to RGB
				int h = H/60;//integer division
				float f = (float)H/60. - h;
				float p = V*(1-S);
				float q = V*(1-S*f);
				float t = V*(1-S*(1-f));

				if(h==0 || h==6){
					R=V;
					G=t;
					B=p;
				}
				else if(h==1){
					R=q;
					G=V;
					B=p;
				}
				else if(h==2){
					R=p;
					G=V;
					B=t;
				}
				else if(h==3){
					R=p;
					G=q;
					B=V;
				}
				else if(h==4){
					R=t;
					G=p;
					B=V;
				}
				else if(h==5){
					R=V;
					G=p;
					B=q;
				}

				//set
				float fac=100;
				plainElevationMapColored_(row, col)[0] = B*fac;
				plainElevationMapColored_(row, col)[1] = G*fac;
				plainElevationMapColored_(row, col)[2] = R*fac;
			}
		}
	}

	//"smooth" image
	//cv::erode(plainElevationMapColored_, plainElevationMap_, cv::Mat());
	//cv::dilate(plainElevationMapColored_, plainElevationMap_, cv::Mat());

	//cv::medianBlur(plainElevationMapColored_, plainElevationMap_, 5);

	//overlay images
	#pragma omp parallel for
	for(int row=0; row < realImg_.rows; ++row)
	{
		for(int col=0; col < realImg_.cols; ++col)
		{
			float sum;

			sum = realImg_(row, col)[0] + plainElevationMapColored_(row, col)[0];
			sum = sum > 255 ? 255 : sum;
			realImg_(row, col)[0] = sum;

			sum = realImg_(row, col)[1] + plainElevationMapColored_(row, col)[1];
			sum = sum > 255 ? 255 : sum;
			realImg_(row, col)[1] = sum;

			sum = realImg_(row, col)[2] + plainElevationMapColored_(row, col)[2];
			sum = sum > 255 ? 255 : sum;
			realImg_(row, col)[2] = sum;
		}
	}
}
