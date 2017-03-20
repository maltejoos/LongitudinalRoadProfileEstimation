/*
 * KalmanFilter.h
 *
 *  Created on: 19.07.2011
 *      Author: joos
 */

#include <opencv/cv.h>

#ifndef KALMANFILTER_H_
#define KALMANFILTER_H_

class KalmanFilter
{
	private:

		double processNoiseScalar;				//factor for Q
		double measurementNoiseScalar;			//factor for R

		cv::Mat R; //observation noise covariance
		cv::Mat R_inv; //inverted observation noise covariance (can be precomputed since noise is uncorrelated)

		cv::Mat x; //state vector
		cv::Mat Q; //process noise
		cv::Mat P; //state covariance

		//H*x=z
		cv::Mat H; //observer matrix
		cv::Mat z; //measurements

		//variables for predict
		float b; 			//base width
		float f; 			//focal length
		int imagecolumns; 	//width of image
		int imagerows;		//height of image
		int imagecenter; 	//center of rows


		bool constantPrediction;

		void predict(cv::Mat& transformationMatrix_f);
		void update(cv::Mat& observerMatrix_d, cv::Mat& measurements_d);


	public:

		cv::Mat spline;
		cv::Mat splineResampled;

		// uncorrelated covariances assumed. All matrices are created by unity matrices and just multiplied with their scalar factors.
		KalmanFilter(cv::Mat stateVectorInit_d, double covarianceOfStateVectorInitScalar, double processNoiseScalar, double measurementNoiseScalar, double baseWidth, double focalLength, int imageRows, int imageColumns);

		// compute kalman step (predict and update)
		//		transformationMatrix_f	:	image transformation matrix from last frame (cf. libviso2)
		//		observerMatrix_d		:	(weighted) observer matrix
		//		measurements_d			:	(weighted) measurements
		void kalmanStep(cv::Mat& transformationMatrix_f, cv::Mat& observerMatrix_d, cv::Mat& measurements_d);

		//reset Kalman Filter
		void reset(cv::Mat stateVectorInit_d, double covarianceOfStateVectorInitScalar);

		// get estimated deBoor points
		void getStateVector(cv::Mat& stateVector_d);

		// get sample of estimated spline (overall result after predict() and update())
		void getSplineSample(cv::Mat& splineSample_i);
};

#endif /* KALMANFILTER_H_ */
