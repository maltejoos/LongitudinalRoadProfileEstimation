/*
 * ObservationEquation.cpp
 *
 *  Created on: 19.07.2011
 *      Author: joos
 */

#include <iostream>
#include "ObservationEquation.h"
#include "CubicBSpline.h"

#define ZERO_CMP 0.00001

int ObservationEquation::countValidPixels(cv::Mat& imgsrc)
{
	int nop=0;
	//for each pixel of imgsrc
	#pragma omp parallel for shared(nop)
	for(int row=0; row<imgsrc.rows; ++row)
	{
		for(int col=0; col<imgsrc.cols; ++col)
		{
			if(imgsrc.at<float>(row, col) > ZERO_CMP){
				#pragma omp atomic
				++nop;
			}
		}
	}

	return nop;
}

ObservationEquation::ObservationEquation(cv::Mat& imgsrc, int numberOfSplines)
{
	int numberOfDBP = numberOfSplines+3;
	int maxModelDisp = imgsrc.cols;
	double knotDistance = (double)maxModelDisp/(double)numberOfSplines;

	int numberOfMeasurements = countValidPixels(imgsrc);

	H = cv::Mat::zeros(numberOfMeasurements, numberOfDBP, CV_64FC1);
	z.create(numberOfMeasurements, 1, CV_64FC1);

	int counter=0;

	bool isSingular = true;

	//for each pixel of imgsrc
	#pragma omp parallel for
	for(int row=0; row<imgsrc.rows; ++row)
	{
		for(int col=0; col<imgsrc.cols; ++col)
		{
			if(imgsrc.at<float>(row, col) <= ZERO_CMP){
				continue;
			}

			double disp = col;
			int splinePart =  disp/knotDistance; //first part is referenced by 0

			if(splinePart == 0){
				isSingular=false;
			}

			//calculate curve parameter for this spline part t e[0,1)
			double t = disp/knotDistance-splinePart;

			//get the B-Spline base values for this parameter
			cv::Mat baseVals;
			CubicBSpline::CUBaseFunctions(t, baseVals);

			#pragma omp critical
			{
				//write matrix H and weight measurement with intensity
				H.at<double>(counter, splinePart) = baseVals.at<double>(0,0)*imgsrc.at<float>(row, col);
				H.at<double>(counter, splinePart+1) = baseVals.at<double>(0,1)*imgsrc.at<float>(row, col);
				H.at<double>(counter, splinePart+2) = baseVals.at<double>(0,2)*imgsrc.at<float>(row, col);
				H.at<double>(counter, splinePart+3) = baseVals.at<double>(0,3)*imgsrc.at<float>(row, col);

				//std::cout << H.at<double>(counter, 0) << " " <<H.at<double>(counter, 1) << " " <<H.at<double>(counter, 2) << " " <<H.at<double>(counter, 3) << " " <<H.at<double>(counter, 4) << " " <<H.at<double>(counter, 5) << " " <<H.at<double>(counter, 6) << " " <<H.at<double>(counter, 7) << std::endl;

				//write weighted measurement to vector z
				z.at<double>(counter, 0) = row*imgsrc.at<float>(row, col);

				++counter;
			}
		}
	}

	if(isSingular){
		std::cout << "Singularity in observation equation!" << std::endl;
	}

	//Should not be reached
	if(counter != numberOfMeasurements){
		std::cout << "Error in Observationequation: Measurement vector has not expected size!" << std::endl;
	}
}
