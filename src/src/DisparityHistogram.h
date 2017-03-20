#include <opencv/cv.h>

struct DisparityHistogram
{	
	// create (v|u)Disparity histogram. SSE optimized
	//		disparityImage_f	:	input disparity image. Attention: image has to be thresholded to maxDisparity.
	//		dest_Histogram_i	:	output histogram
	//		maxDisparity		: 	the maximal disparity in the output histogram (i.e. the size of the created histogram)
	// Note: max disparity of disparityImage_f <= maxDisp. Threshold input image before.
	static void calculateVDisparity(cv::Mat& disparityImage_f, cv::Mat& dest_Histogram_i, int maxDisparity);
	static void calculateUDisparity(cv::Mat& disparityImage_f, cv::Mat& dest_Histogram_i, int maxDisparity);
	
	// remove (set to zero) pixels from disparityImage if corresponding value in uDisparity is bigger than threshold
	//		disparityImage_f	:	input disparity image
	//		uDisparity_i		: 	uDisparity histogram of disparityImage_f
	//		dest_DisparityImage_f:	filtered disparity image
	//		threshold			:	filter threshold
	static void filterObstaclesFromUD(cv::Mat& disparityImage_f, cv::Mat& uDisparity_i, cv::Mat& dest_DisparityImage_f, int threshold);

	static float estimateRollAngle(cv::Mat& disparityImage_f, int maxDisparity, float minimumAngle, float maximumAngle, float angleStep);
};
