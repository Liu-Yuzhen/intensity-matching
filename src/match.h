#ifndef MATCH_H
#define MATCH_H
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class Matcher
{
public:
    enum
    {
	CORR = 0,
	COV = 1,
	SAD = 2,
	SSD = 3,
	NCC = 4,
	LS = 5
    };

    static void match(const cv::Mat& img1, const cv::Mat& img2,
		      const std::vector<cv::KeyPoint>& keypoints1,
		      std::vector<cv::KeyPoint>& keypoints2,
		      std::vector<cv::DMatch>& matches, int wsize,
		      int method = 0, int epipolar = INT_MAX, float overlap = 1.0);
};

#endif /* MATCH_H */
