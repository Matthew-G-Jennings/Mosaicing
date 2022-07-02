#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "Timer.h"
#include <math.h>


cv::Mat fix;

cv::Mat translationMatrix(double dx, double dy) {
	cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
	T.at<double>(0, 2) = dx;
	T.at<double>(1, 2) = dy;
	return T;
}

cv::Mat resizeTarget(cv::Mat original, cv::Mat second, cv::Mat T) {

	double umax = original.size().width;
	double umin = 0;
	double vmax = original.size().height;
	double vmin = 0;

	cv::Mat c1_(3, 1, CV_64F);
	cv::Mat c2_(3, 1, CV_64F);
	cv::Mat c3_(3, 1, CV_64F);
	cv::Mat c4_(3, 1, CV_64F);

	c1_.at<double>(0, 0) = 0;
	c1_.at<double>(1, 0) = 0;
	c1_.at<double>(2, 0) = 1;

	c2_.at<double>(0, 0) = second.size().width;
	c2_.at<double>(1, 0) = 0;
	c2_.at<double>(2, 0) = 1;

	c3_.at<double>(0, 0) = second.size().width;
	c3_.at<double>(1, 0) = second.size().height;
	c3_.at<double>(2, 0) = 1;

	c4_.at<double>(0, 0) = 0;
	c4_.at<double>(1, 0) = second.size().height;
	c4_.at<double>(2, 0) = 1;

	cv::Mat c1 = T * c1_;
	c1.at<double>(0, 0) = c1.at<double>(0, 0) / c1.at<double>(2, 0);
	c1.at<double>(1, 0) = c1.at<double>(1, 0) / c1.at<double>(2, 0);
	c1.at<double>(2, 0) = c1.at<double>(2, 0) / c1.at<double>(2, 0);
	if (c1.at<double>(0, 0) < umin) {
		umin = c1.at<double>(0, 0);
	}
	else if (c1.at<double>(0, 0) > umax) {
		umax = c1.at<double>(0, 0);
	}
	if (c1.at<double>(1, 0) < vmin) {
		vmin = c1.at<double>(1, 0);
	}
	else if (c1.at<double>(1, 0) > vmax) {
		vmax = c1.at<double>(1, 0);
	}

	cv::Mat c2 = T * c2_;
	c2.at<double>(0, 0) = c2.at<double>(0, 0) / c2.at<double>(2, 0);
	c2.at<double>(1, 0) = c2.at<double>(1, 0) / c2.at<double>(2, 0);
	c2.at<double>(2, 0) = c2.at<double>(2, 0) / c2.at<double>(2, 0);
	if (c2.at<double>(0, 0) < umin) {
		umin = c2.at<double>(0, 0);
	}
	else if (c2.at<double>(0, 0) > umax) {
		umax = c2.at<double>(0, 0);
	}
	if (c2.at<double>(1, 0) < vmin) {
		vmin = c2.at<double>(1, 0);
	}
	else if (c2.at<double>(1, 0) > vmax) {
		vmax = c2.at<double>(1, 0);
	}

	cv::Mat c3 = T * c3_;
	c3.at<double>(0, 0) = c3.at<double>(0, 0) / c3.at<double>(2, 0);
	c3.at<double>(1, 0) = c3.at<double>(1, 0) / c3.at<double>(2, 0);
	c3.at<double>(2, 0) = c3.at<double>(2, 0) / c3.at<double>(2, 0);
	if (c3.at<double>(0, 0) < umin) {
		umin = c3.at<double>(0, 0);
	}
	else if (c3.at<double>(0, 0) > umax) {
		umax = c3.at<double>(0, 0);
	}
	if (c3.at<double>(1, 0) < vmin) {
		vmin = c3.at<double>(1, 0);
	}
	else if (c3.at<double>(1, 0) > vmax) {
		vmax = c3.at<double>(1, 0);
	}

	cv::Mat c4 = T * c4_;
	c4.at<double>(0, 0) = c4.at<double>(0, 0) / c4.at<double>(2, 0);
	c4.at<double>(1, 0) = c4.at<double>(1, 0) / c4.at<double>(2, 0);
	c4.at<double>(2, 0) = c4.at<double>(2, 0) / c4.at<double>(2, 0);
	if (c4.at<double>(0, 0) < umin) {
		umin = c4.at<double>(0, 0);
	}
	else if (c4.at<double>(0, 0) > umax) {
		umax = c4.at<double>(0, 0);
	}
	if (c4.at<double>(1, 0) < vmin) {
		vmin = c4.at<double>(1, 0);
	}
	else if (c4.at<double>(1, 0) > vmax) {
		vmax = c4.at<double>(1, 0);
	}
	/*
	std::cout << std::endl;
	std::cout << c1 << std::endl;
	std::cout << std::endl;
	std::cout << c2 << std::endl;
	std::cout << std::endl;
	std::cout << c3 << std::endl;
	std::cout << std::endl;
	std::cout << c4 << std::endl;
	std::cout << std::endl;

	std::cout << umin << " " << umax << " " << vmin << " " << vmax << std::endl;
	*/
	fix = translationMatrix(-umin, -vmin);

	return cv::Mat(vmax - vmin, umax - umin, CV_8UC3);

}

cv::Mat scaleMatrix(double s) {
	cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
	T.at<double>(0, 0) = s;
	T.at<double>(1, 1) = s;
	return T;
}

void homoComp(cv::Mat H_, cv::Mat H) {
	
	double squaresum = 0.0;
	double temp;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			temp = H_.at<double>(i, j) - H.at<double>(i, j);
			squaresum += temp * temp;
		}
	}
	squaresum = sqrt(squaresum);

	std::cout << "Frobenius norm: " << squaresum << std::endl;


}

cv::Mat homoGen(cv::Mat image1, cv::Mat image2, double scale, double RSThresh) {

	Timer myTimer;
	
	//FEATURE DETECION START

	cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
	std::vector<cv::KeyPoint> keypoints1;
	cv::Mat descriptors1;
	std::vector<cv::KeyPoint> keypoints2;
	cv::Mat descriptors2;

	myTimer.reset();
	detector->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	std::cout << "Found " << keypoints1.size() << " features in first image which took " << myTimer.read() << " seconds" << std::endl;
	cv::Mat kptImage1;
	cv::drawKeypoints(image1, keypoints1, kptImage1, cv::Scalar(0, 255, 0),
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	myTimer.reset();
	detector->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
	std::cout << "Found " << keypoints2.size() << " features in second image which took " << myTimer.read() << " seconds" << std::endl;
	cv::Mat kptImage2;
	cv::drawKeypoints(image2, keypoints2, kptImage2, cv::Scalar(0, 255, 0),
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//FEATURE DETECTION END

	//FEATURE MATCHING START
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();

	myTimer.reset();
	std::vector<std::vector<cv::DMatch>> matches;
	matcher->knnMatch(descriptors1, descriptors2, matches, 2);

	std::vector<cv::DMatch> goodMatches;
	std::vector<cv::Point2f> goodPts1, goodPts2;

	for (const auto& match : matches) {
		if (match[0].distance < 0.8 * match[1].distance) {
			goodMatches.push_back(match[0]);
			goodPts1.push_back(keypoints1[match[0].queryIdx].pt);
			goodPts2.push_back(keypoints2[match[0].trainIdx].pt);
		}
	}
	std::cout << "Matching took " << myTimer.read() << " seconds resulting in " << goodMatches.size() << " good matches out of " << matches.size() << std::endl;
	//FEATURE MATCHING END

	//HOMOGRAPHY ESTIMATION START
	std::vector<unsigned char> inliers;
	cv::Mat H = cv::findHomography(goodPts2, goodPts1, inliers, cv::RANSAC, RSThresh);
	//HOMOGRAPHY ESTIMATION END


	H.at<double>(0, 2) = H.at<double>(0, 2) / scale;
	H.at<double>(1, 2) = H.at<double>(1, 2) / scale;
	H.at<double>(2, 0) = H.at<double>(2, 0) * scale;
	H.at<double>(2, 1) = H.at<double>(2, 1) * scale;

	std::cout << std::endl;
	std::cout << H;
	std::cout << std::endl;
	
	return H;
}

int main(int argc, char* argv[]) {

	cv::Mat image1 = cv::imread("../image1.jpeg");
	cv::Mat image2 = cv::imread("../image2.jpeg");

	if (image1.empty()) {
		std::cerr << "Could not load image from image1.jpg" << std::endl;
		return -1;
	}
	if (image2.empty()) {
		std::cerr << "Could not load image from image2.jpg" << std::endl;
		return -1;
	}

	cv::Mat GH = homoGen(image1, image2, 1, 3);

	double scale = 1;
	/*
	cv::warpPerspective(image1, image1, scaleMatrix(scale), image1.size()/(int)(1/scale),
		cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);
	cv::warpPerspective(image2, image2, scaleMatrix(scale), image2.size()/(int)(1/scale),
		cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);
		*/
	std::cout << image1.size() << std::endl;

	cv::Mat H = homoGen(image1, image2, 1, 10);

	homoComp(GH, H);
	

	cv::Mat image3 = cv::imread("C:/COSC300/342A1/E/image1.jpeg");
	cv::Mat image4 = cv::imread("C:/COSC300/342A1/E/image2.jpeg");
	

	cv::Mat mosaic = resizeTarget(image3, image4, H);


	cv::warpPerspective(image3, mosaic, fix, mosaic.size(),
		cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);

	cv::warpPerspective(image4, mosaic, fix*H, mosaic.size(),
		cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);

	
	cv::warpPerspective(mosaic, mosaic, scaleMatrix(0.25), mosaic.size()/4,
		cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);
	
	//cv::Mat matchImg;
	//cv::drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, matchImg);
	cv::namedWindow("Mosaic");
	cv::imshow("Mosaic", mosaic);

	/*
	cv::namedWindow("Display 1");
	cv::imshow("Display 1", kptImage1);
	cv::imshow("Display 2", kptImage2);
	*/
	cv::waitKey();

	return 0;
}