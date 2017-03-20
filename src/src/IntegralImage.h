#include <opencv/cv.h>

struct IntegralImage
{
	// calculate (u|v) integral image. SSE optimized
	static void vIntegralImage(cv::Mat& src_Image_f, cv::Mat& dest_vIntegralImage_f);
	static void uIntegralImageR(cv::Mat& src_Image_f, cv::Mat& dest_uIntegralImage_f);

	// calculate weights for v-integral image. SSE optimized
	// weight(u,v) = integral(u, 0..v)/integral(u, 0..v_max). Weights e [0, 1]
	static void vWeight(cv::Mat& src_Image_f, cv::Mat& dest_weightImage_f);

	// calculate reverse weights for u-integral image. SSE optimized
	// weight(u,v) = integral(u..u_max, v)/integral(0..u_max, v). Weights e [0, 1]
	static void uWeightR(cv::Mat& src_Image_f, cv::Mat& dest_weightImage_f);

	// map weight image to percentage
	// weight(0) => 0;	weight(percentage) => 1;	weight(1) => 0. In between linear interpolation.
	static void mapWeightPC(cv::Mat& weightImg_f, cv::Mat& dest_mappedImg_f, float percentage);

	// elementwise multiplication of weightImage_f and src_Image_f, "times" times
	static void applyWeight(cv::Mat& src_Image_f, cv::Mat& weightImage_f, cv::Mat& dest_weightedImage, int times);
};
