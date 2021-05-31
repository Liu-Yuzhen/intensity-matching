/******************************************************************************
 * File:             main.cpp
 *
 * Author:           liuyuzhen
 * Created:          21/05/27
 * Description:      digit photogrammetry
 *****************************************************************************/

#include <iostream>

#include "match.h"

static void show_usage(const std::string name)
{
    std::cout << "Usage: " << name << "\nOptions:\n"
	      << "\t-p1 <file>,\tspecify img1 path\n"
	      << "\t-p2 <file>,\tspecify img2 path\n"
	      << "\t-o <file>, \tplace the output image to <file>\n"
	      << "\t-m <int>, \t0:CORR, 1:COV, 2:SAD, 3:SSD, 4:NCC, 5:LS\n"
	      << "\t-t <float>, \tspecify fast corner threshold\n"
	      << "\t-s <float>, \tscale the image\n"
	      << "\t-e <int>, \tepipolar constraint\n"
	      << "\t-l <float>, \toverlap\n"
	      << "\t-w <int>,\tspecify window size\n";
}

static int parse_command(int argc, char* argv[], int& method, float& threshold,
			 int& wsize, float& scale, std::string& path1,
			 std::string& path2, std::string& path_dst, int& epi, float& overlap)
{
    for (int i = 1; i < argc; i++)
	{
	    std::string arg = argv[i];
	    if (arg == "-h" || arg == "--help")
		{
		    show_usage(argv[0]);
		    return -1;
		}
	    else if (arg == "-l")
		{
		    if (i + 1 < argc)
			{
			    overlap = std::atof(argv[i + 1]);
			}
		    else
			{
			    show_usage(argv[0]);
			    return -1;
			}
		}
	    else if (arg == "-e")
		{
		    if (i + 1 < argc)
			{
			    epi = std::atoi(argv[i + 1]);
			}
		    else
			{
			    show_usage(argv[0]);
			    return -1;
			}
		}
	    else if (arg == "-s" || arg == "--scale")
		{
		    if (i + 1 < argc)
			{
			    scale = std::atof(argv[i + 1]);
			}
		    else
			{
			    show_usage(argv[0]);
			    return -1;
			}
		}
	    else if (arg == "-m" || arg == "--method")
		{
		    if (i + 1 < argc)
			{
			    method = std::atoi(argv[i + 1]);
			}
		    else
			{
			    show_usage(argv[0]);
			    return -1;
			}
		}
	    else if (arg == "-t" || arg == "--threshold")
		{
		    if (i + 1 < argc)
			{
			    threshold = std::atof(argv[i + 1]);
			}
		    else
			{
			    show_usage(argv[0]);
			    return -1;
			}
		}
	    else if (arg == "-p1" || arg == "--path1")
		{
		    if (i + 1 < argc)
			{
			    path1 = argv[i + 1];
			}
		    else
			{
			    show_usage(argv[0]);
			    return -1;
			}
		}
	    else if (arg == "-p2" || arg == "--path2")
		{
		    if (i + 1 < argc)
			{
			    path2 = argv[i + 1];
			}
		    else
			{
			    show_usage(argv[0]);
			    return -1;
			}
		}
	    else if (arg == "-o")
		{
		    if (i + 1 < argc)
			{
			    path_dst = argv[i + 1];
			}
		    else
			{
			    show_usage(argv[0]);
			    return -1;
			}
		}
	    else if (arg == "-w")
		{
		    if (i + 1 < argc)
			{
			    wsize = std::atoi(argv[i + 1]);
			    if (wsize % 2 != 1)
				{
				    std::cout << "window size must be odd"
					      << std::endl;
				    return -1;
				}
			}
		    else
			{
			    show_usage(argv[0]);
			    return -1;
			}
		}
	}

    if (path1 == "" || path2 == "")
	{
	    std::cout << "Please specify image path" << std::endl;
	    show_usage(argv[0]);
	    return -1;
	}

    return 1;
}

int main(int argc, char** argv)
{
    int method = 0;
    float threshold = 150.0;
    int win_size = 31;
    float scale = 0.6;
    int epi = 1e8;
	float overlap = 1.0;

    std::string path1, path2;
    std::string path_dst;

    if (parse_command(argc, argv, method, threshold, win_size, scale, path1,
		      path2, path_dst, epi, overlap) == -1)
	{
	    return -1;
	}

	overlap = (overlap > 0.0 && overlap <= 1.0) ? overlap: 1;

    std::cout << "Params:\n"
	      << "method:\t" << method << "\nthreshold:\t" << threshold
	      << "\nwindow size:\t" << win_size << "\nscale:\t" << scale
	      << "\nimg1:\t" << path1 << "\nimg2:\t" << path2
		  << "\nepipolar constraint:\t" << epi 
		  << "\noverlap:\t" << overlap 
	      << "\ndestination:\t" << path_dst << std::endl;

    cv::Mat img1 = cv::imread(path1);
    cv::Mat img2 = cv::imread(path2);

    if (img1.empty())
	{
	    std::cout << "Failed to load:" << path1 << std::endl;
	    return -1;
	}

    if (img2.empty())
	{
	    std::cout << "Failed to load:" << path2 << std::endl;
	    return -1;
	}

    // resize
    cv::resize(img1, img1, cv::Size(0, 0), scale, scale);
    cv::resize(img2, img2, cv::Size(0, 0), scale, scale);

    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1, img1_gray, CV_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, CV_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<cv::DMatch> matches;

    cv::Ptr<cv::Feature2D> f2d = cv::FastFeatureDetector::create(threshold);
    f2d->detect(img1_gray, keypoints1);

    Matcher::match(img1_gray, img2_gray, keypoints1, keypoints2, matches,
		   win_size, method, epi, overlap);

    std::cout << "total keypoints: " << keypoints2.size() << std::endl;

    cv::Mat img_show;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_show);

    if (path_dst != "")
	{
	    cv::imwrite(path_dst, img_show);
	}

    cv::imshow("matches", img_show);
    cv::waitKey(0);
    return 0;
}
