/******************************************************************************
 * File:             match.cpp
 *
 * Author:           liuyuzhen
 * Created:          21/05/28
 * Description:      digital photogrammetry
 *****************************************************************************/

#include "match.h"

#include <iostream>

void matchSAD(const cv::Mat& img1, const cv::Mat& img2,
	      const std::vector<cv::KeyPoint>& keypoints1,
	      std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches,
	      int wsize, int epipolar, float overlap)
{
    int half = (wsize - 1) >> 1;
    int count = 0;
    int xl = (1 - overlap) * img1.cols;
    int xr = overlap * img1.cols;
    xl = xl > half ? xl : half;
    xr = xr < img2.cols - half ? xr : img2.cols - half;
    for (int i = 0; i < keypoints1.size(); ++i)
	{
	    int x = keypoints1[i].pt.x;
	    if (x < xl) continue;

	    int y = keypoints1[i].pt.y;

	    // check for boundry constraint
	    if (x - half < 0 || x + half >= img1.cols) continue;
	    if (y - half < 0 || y + half >= img1.rows) continue;

	    float min_dist = FLT_MAX;
	    cv::Point2f best_pt(-1, -1);

	    // epipolar constraint: search in [y - kszie, y
	    // + ksize]
	    int yl = half < y - epipolar ? y - epipolar : half;
	    int yh = img2.rows - half > epipolar + y ? epipolar + y : img2.rows - half;

	    for (int r = yl; r < yh; ++r)
		{
		    for (int c = half; c < xr; ++c)
			{
			    int sum = 0;
			    for (int kr = -half; kr <= half; ++kr)
				{
				    for (int kc = -half; kc <= half; ++kc)
					{
					    int gray1 = static_cast<int>(
						img1.at<uchar>(y + kr, x + kc));
					    int gray2 = static_cast<int>(
						img2.at<uchar>(r + kr, c + kc));

					    sum += std::abs(gray1 - gray2);
					}
				}
			    if (sum < min_dist)
				{
				    min_dist = sum;
				    best_pt.y = r;
				    best_pt.x = c;
				}
			}
		}

	    cv::DMatch mc;
	    mc.trainIdx = count++;
	    mc.queryIdx = i;
	    mc.distance = min_dist;

	    cv::KeyPoint kp;
	    kp.pt = best_pt;
	    keypoints2.push_back(kp);
	    matches.push_back(mc);
	}
}

void matchSSD(const cv::Mat& img1, const cv::Mat& img2,
	      const std::vector<cv::KeyPoint>& keypoints1,
	      std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches,
	      int wsize, int epipolar, float overlap)
{
    int half = (wsize - 1) >> 1;
    int count = 0;
    int xl = (1 - overlap) * img1.cols;
    int xr = overlap * img1.cols;
    xl = xl > half ? xl : half;
    xr = xr < img2.cols - half ? xr : img2.cols - half;
    for (int i = 0; i < keypoints1.size(); ++i)
	{
	    int x = keypoints1[i].pt.x;
	    if (x < xl) continue;
	    int y = keypoints1[i].pt.y;

	    // check for boundry constraint
	    if (x - half < 0 || x + half >= img1.cols) continue;
	    if (y - half < 0 || y + half >= img1.rows) continue;

	    // epipolar constraint: search in [y - kszie, y
	    // + ksize]
	    int yl = half < y - epipolar ? y - epipolar : half;
	    int yh = img2.rows - half > epipolar + y ? epipolar + y : img2.rows - half;

	    int min_dist = INT_MAX;
	    cv::Point2f best_pt(-1, -1);
	    for (int r = yl; r < yh; ++r)
		{
		    for (int c = half; c < xr; ++c)
			{
			    int sum = 0;
			    for (int kr = -half; kr <= half; ++kr)
				{
				    for (int kc = -half; kc <= half; ++kc)
					{
					    int gray1 = static_cast<int>(
						img1.at<uchar>(y + kr, x + kc));
					    int gray2 = static_cast<int>(
						img2.at<uchar>(r + kr, c + kc));

					    int diff = gray1 - gray2;
					    sum += diff * diff;
					}
				}
			    if (sum < min_dist)
				{
				    min_dist = sum;
				    best_pt.y = r;
				    best_pt.x = c;
				}
			}
		}

	    cv::DMatch mc;
	    mc.trainIdx = count++;
	    mc.queryIdx = i;
	    mc.distance = min_dist;

	    cv::KeyPoint kp;
	    kp.pt = best_pt;
	    keypoints2.push_back(kp);
	    matches.push_back(mc);
	}
}

void matchCOV(const cv::Mat& img1, const cv::Mat& img2,
	      const std::vector<cv::KeyPoint>& keypoints1,
	      std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches,
	      int wsize, int epipolar, float overlap)
{
    int half = (wsize - 1) >> 1;
    int n = wsize * wsize;
    int count = 0;
    int xl = (1 - overlap) * img1.cols;
    int xr = overlap * img1.cols;
    xl = xl > half ? xl : half;
    xr = xr < img2.cols - half ? xr : img2.cols - half;
    for (int i = 0; i < keypoints1.size(); ++i)
	{
	    int x = keypoints1[i].pt.x;
	    if (x < xl) continue;
	    int y = keypoints1[i].pt.y;

	    // check for boundry constraint
	    if (x - half < 0 || x + half >= img1.cols) continue;
	    if (y - half < 0 || y + half >= img1.rows) continue;

	    int yl = half < y - epipolar ? y - epipolar : half;
	    int yh = img2.rows - half > epipolar + y ? epipolar + y : img2.rows - half;

	    float max_response = FLT_MIN;
	    cv::Point2f best_pt(-1, -1);
	    for (int r = yl; r < yh; ++r)
		{
		    for (int c = half; c < xr; ++c)
			{
			    int conv = 0.0;
			    int sum1 = 0.0;
			    int sum2 = 0.0;
			    for (int kr = -half; kr <= half; ++kr)
				{
				    for (int kc = -half; kc <= half; ++kc)
					{
					    int gray1 = static_cast<int>(
						img1.at<uchar>(y + kr, x + kc));
					    int gray2 = static_cast<int>(
						img2.at<uchar>(r + kr, c + kc));

					    conv += gray1 * gray2;
					    sum1 += gray1;
					    sum2 += gray2;
					}
				}
			    float response = static_cast<float>(conv) - 
					static_cast<float>(sum1) * static_cast<float>(sum2) / n;
			    if (response > max_response)
				{
				    max_response = response;
				    best_pt.y = r;
				    best_pt.x = c;
				}
			}
		}

	    cv::DMatch mc;
	    mc.trainIdx = count++;
	    mc.queryIdx = i;

	    cv::KeyPoint kp;
	    kp.pt = best_pt;
	    keypoints2.push_back(kp);
	    matches.push_back(mc);
	}
}

void matchCORR(const cv::Mat& img1, const cv::Mat& img2,
	       const std::vector<cv::KeyPoint>& keypoints1,
	       std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches,
	       int wsize, int epipolar, float overlap)
{
    int half = (wsize - 1) >> 1;
    int count = 0;
    int n = wsize * wsize;
    int xl = (1 - overlap) * img1.cols;
    int xr = overlap * img1.cols;
    xl = xl > half ? xl : half;
    xr = xr < img2.cols - half ? xr : img2.cols - half;
    for (int i = 0; i < keypoints1.size(); ++i)
	{
	    int x = keypoints1[i].pt.x;
	    if (x < xl) continue;
	    int y = keypoints1[i].pt.y;

	    // check for boundry constraint
	    if (x - half < 0 || x + half >= img1.cols) continue;
	    if (y - half < 0 || y + half >= img1.rows) continue;

	    int yl = half < y - epipolar ? y - epipolar : half;
	    int yh = img2.rows - half > epipolar + y ? epipolar + y : img2.rows - half;

	    float max_response = FLT_MIN;
	    cv::Point2f best_pt(-1, -1);
	    for (int r = yl; r < yh; ++r)
		{
		    for (int c = half; c < xr; ++c)
			{
			    int conv = 0.0;
			    int ss1 = 0.0;
			    int ss2 = 0.0;
			    int sum1 = 0.0;
			    int sum2 = 0.0;
			    for (int kr = -half; kr <= half; ++kr)
				{
				    for (int kc = -half; kc <= half; ++kc)
					{
					    int gray1 = static_cast<int>(
						img1.at<uchar>(y + kr, x + kc));
					    int gray2 = static_cast<int>(
						img2.at<uchar>(r + kr, c + kc));

					    conv += gray1 * gray2;
					    ss1 += gray1 * gray1;
					    ss2 += gray2 * gray2;
					    sum1 += gray1;
					    sum2 += gray2;
					}
				}

			    float response =
				(static_cast<double>(n) * static_cast<double>(conv) -
				 static_cast<double>(sum1) * static_cast<double>(sum2));
			    response /= std::sqrt(
				(static_cast<double>(n) * static_cast<double>(ss1) -
				 static_cast<double>(sum1) * sum1) *
				(static_cast<double>(n) * static_cast<double>(ss2) -
				 static_cast<double>(sum2) * static_cast<double>(sum2)));
			    if (response > max_response)
				{
				    max_response = response;
				    best_pt.y = r;
				    best_pt.x = c;
				}
			}
		}

	    cv::DMatch mc;
	    mc.trainIdx = count++;
	    mc.queryIdx = i;

	    cv::KeyPoint kp;
	    kp.pt = best_pt;
	    keypoints2.push_back(kp);
	    matches.push_back(mc);
	}
}

void matchNCC(const cv::Mat& img1, const cv::Mat& img2,
	      const std::vector<cv::KeyPoint>& keypoints1,
	      std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches,
	      int wsize, int epipolar, float overlap)
{
    int half = (wsize - 1) >> 1;
    int count = 0;
    int xl = (1 - overlap) * img1.cols;
    int xr = overlap * img1.cols;
    xl = xl > half ? xl : half;
    xr = xr < img2.cols - half ? xr : img2.cols - half;
    for (int i = 0; i < keypoints1.size(); ++i)
	{
	    int x = keypoints1[i].pt.x;
	    if (x < xl) continue;
	    int y = keypoints1[i].pt.y;

	    // check for boundry constraint
	    if (x - half < 0 || x + half >= img1.cols) continue;
	    if (y - half < 0 || y + half >= img1.rows) continue;

	    int yl = half < y - epipolar ? y - epipolar : half;
	    int yh = img2.rows - half > epipolar + y ? epipolar + y : img2.rows - half;

	    float max_response = FLT_MIN;
	    cv::Point2f best_pt(-1, -1);
	    for (int r = yl; r < yh; ++r)
		{
		    for (int c = half; c < xr; ++c)
			{
			    int conv = 0.0;
			    int ss1 = 0.0;
			    int ss2 = 0.0;
			    for (int kr = -half; kr <= half; ++kr)
				{
				    for (int kc = -half; kc <= half; ++kc)
					{
					    int gray1 = static_cast<int>(
						img1.at<uchar>(y + kr, x + kc));
					    int gray2 = static_cast<int>(
						img2.at<uchar>(r + kr, c + kc));

					    conv += gray1 * gray2;
					    ss1 += gray1 * gray1;
					    ss2 += gray2 * gray2;
					}
				}
			    float response = static_cast<float>(conv) /
				   	std::sqrt(static_cast<float>(ss1) * static_cast<float>(ss2));
			    if (response > max_response)
				{
				    max_response = response;
				    best_pt.y = r;
				    best_pt.x = c;
				}
			}
		}

	    cv::DMatch mc;
	    mc.trainIdx = count++;
	    mc.queryIdx = i;

	    cv::KeyPoint kp;
	    kp.pt = best_pt;
	    keypoints2.push_back(kp);
	    matches.push_back(mc);
	}
}

void computeGradient(const cv::Mat& img, cv::Mat& gradientX, cv::Mat& gradientY)
{
    gradientX = cv::Mat::zeros(img.rows, img.cols, CV_32S);
    gradientY = cv::Mat::zeros(img.rows, img.cols, CV_32S);

    for (int i = 1; i < img.rows - 1; ++i)
	{
	    for (int j = 1; j < img.cols - 1; ++j)
		{
		    gradientX.at<int>(i, j) =
			(static_cast<int>(img.at<uchar>(i, j + 1)) -
			 static_cast<int>(img.at<uchar>(i, j - 1))) /
			2;
		    gradientY.at<int>(i, j) =
			(static_cast<int>(img.at<uchar>(i + 1, j)) -
			 static_cast<int>(img.at<uchar>(i - 1, j))) /
			2;
		}
	}
}

template <typename T>
T interpolate(const cv::Mat& img, float x, float y)
{
    int x1 = static_cast<int>(x);
    int x2 = x1 + 1;
    int y1 = static_cast<int>(y);
    int y2 = y1 + 1;
    int c11 = static_cast<int>(img.at<T>(y1, x1));
    int c12 = static_cast<int>(img.at<T>(y1, x2));
    int c21 = static_cast<int>(img.at<T>(y2, x1));
    int c22 = static_cast<int>(img.at<T>(y2, x2));
    float gray1 = (x - x1) * c12 + (x2 - x) * c11;
    float gray2 = (x - x1) * c22 + (x2 - x) * c21;
    float gray = gray1 * (y - y1) + gray2 * (y2 - y);

    return std::round(gray);
}

void matchLS(const cv::Mat& img1, const cv::Mat& img2,
	     const std::vector<cv::KeyPoint>& keypoints1,
	     std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches,
	     int wsize, int epipolar, float overlap)
{
    // compute gradient using difference
    cv::Mat gradX, gradY;
    computeGradient(img2, gradX, gradY);

    int n = wsize * wsize;
    int half = (wsize - 1) >> 1;
    half = half >= 2 ? half : 2;
    int xl = (1 - overlap) * img1.cols;
    int xr = overlap * img1.cols;
    xl = xl > half ? xl : half;
    xr = xr < img2.cols - half ? xr : img2.cols - half;

    std::vector<cv::DMatch> matches_tmp;
    matchCORR(img1, img2, keypoints1, keypoints2, matches_tmp, wsize, epipolar, overlap);
    std::vector<bool> is_success = std::vector<bool>(matches_tmp.size(), false);
    for (int i = 0; i < matches_tmp.size(); ++i)
	{
	    cv::Point2f pt1 = keypoints1[matches_tmp[i].queryIdx].pt;
	    int x = pt1.x;
	    if (x < xl) continue;
	    int y = pt1.y;

	    cv::Point2f pt2 = keypoints2[matches_tmp[i].trainIdx].pt;
	    // check for boundry constraint
	    /* if (x - half < 0 || x + half >= img1.cols) continue;
	     * if (y - half < 0 || y + half >= img1.rows) continue; */

	    cv::Mat X = cv::Mat::zeros(8, 1, CV_32F);
	    cv::Mat V;
	    X.at<float>(1, 0) = 1.0;
	    X.at<float>(3, 0) = 1.0;
	    X.at<float>(7, 0) = 1.0;
	    for (int iter = 0; iter < 50; ++iter)
		{
		    cv::Mat B = cv::Mat::zeros(n, 8, CV_32F);
		    cv::Mat L = cv::Mat::zeros(n, 1, CV_32F);
		    // h0. h1. a0, a1, a2,
		    // b0, b1, b2

		    float h0 = X.at<float>(0, 0);
		    float h1 = X.at<float>(1, 0);
		    float a0 = X.at<float>(2, 0);
		    float a1 = X.at<float>(3, 0);
		    float a2 = X.at<float>(4, 0);
		    float b0 = X.at<float>(5, 0);
		    float b1 = X.at<float>(6, 0);
		    float b2 = X.at<float>(7, 0);

		    int row = 0;
		    bool flag = true;
		    for (int kr = -half; kr <= half; ++kr)
			{
			    for (int kc = -half; kc <= half; ++kc)
				{
				    float x_ = a0 + a1 * kc + a2 * kr;
				    float y_ = b0 + b1 * kc + b2 * kr;

				    int gray1 =
					static_cast<int>(img1.at<uchar>(y + kr, x + kc));

				    float x2 = pt2.x + x_;
				    float y2 = pt2.y + y_;

				    if (x2 <= 0.0 || x2 >= img2.cols || y2 <= 0.0 ||
					y2 >= img2.rows)
					{
					    flag = false;
					    break;
					}

				    int gray2 = interpolate<uchar>(img2, x2, y2);

				    int gx = interpolate<int>(gradX, x2, y2);

				    int gy = interpolate<int>(gradY, x2, y2);

				    int diff = gray1 - (h0 + h1 * gray2);

				    B.at<float>(row, 0) = 1;
				    B.at<float>(row, 1) = gray2;
				    B.at<float>(row, 2) = gx;
				    B.at<float>(row, 3) = x_ * gx;
				    B.at<float>(row, 4) = y_ * gx;
				    B.at<float>(row, 5) = gy;
				    B.at<float>(row, 6) = x_ * gy;
				    B.at<float>(row, 7) = y_ * gy;

				    L.at<float>(row, 0) = diff;

				    ++row;
				}
			}

		    cv::Mat Bt = B.t();
		    cv::Mat delta_X = (Bt * B).inv() * Bt * L;
		    X = X + delta_X;

		    cv::Mat change = delta_X.t() * delta_X;
		    if (change.at<float>(0, 0) < 1)
			{
			    /* std::cout << X.t() << std::endl;
			     * cv::Mat V = B * delta_X - L;
			     * std::cout << "residual = " << V.t() << std::endl; */
			    is_success[i] = true;
			    cv::DMatch& mc = matches_tmp[i];
			    cv::Point2f& pt = keypoints2[mc.trainIdx].pt;
			    // modify the coordinate
			    pt.x = pt2.x + X.at<float>(2, 0);
			    pt.y = pt2.y + X.at<float>(5, 0);

			    break;
			}
		    else if (!flag || X.at<float>(2, 0) > 2 || X.at<float>(5, 0) > 2)
			{
			    break;
			}
		}
	}
    for (int i = 0; i < is_success.size(); ++i)
	{
	    if (is_success[i]) matches.push_back(matches_tmp[i]);
	}
}

void Matcher::match(const cv::Mat& img1, const cv::Mat& img2,
		    const std::vector<cv::KeyPoint>& keypoints1,
		    std::vector<cv::KeyPoint>& keypoints2,
		    std::vector<cv::DMatch>& matches, int wsize, int method, int y,
		    float overlap)
{
    switch (method)
	{
	    case CORR:
		matchCORR(img1, img2, keypoints1, keypoints2, matches, wsize, y, overlap);
		break;

	    case COV:
		matchCOV(img1, img2, keypoints1, keypoints2, matches, wsize, y, overlap);
		break;

	    case SAD:
		matchSAD(img1, img2, keypoints1, keypoints2, matches, wsize, y, overlap);
		break;

	    case SSD:
		matchSSD(img1, img2, keypoints1, keypoints2, matches, wsize, y, overlap);
		break;

	    case NCC:
		matchNCC(img1, img2, keypoints1, keypoints2, matches, wsize, y, overlap);
		break;

	    case LS:
		matchLS(img1, img2, keypoints1, keypoints2, matches, wsize, y, overlap);
		break;

	    default:
		matchCORR(img1, img2, keypoints1, keypoints2, matches, wsize, y, overlap);
		break;
	}
}

