/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

// varying parameters: 
// - detectors      {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"}
// - descriptors    {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"}
// - matcherType    {"MAT_BF", "MAT_FLANN"}
// - descriptorType {"DES_BINARY", "DES_HOG"}
// - selectorType   {"SEL_NN", "SEL_KNN"}
void saveReport(std::vector<DetectionInfo> &combinationInfo)
{
    // create and open the .csv file
    fstream reportFile;    
    reportFile.open("SFND_FeatureTracking_Report.csv", std::fstream::in  | std::fstream::out | std::fstream::app);

    if (reportFile) {
        // write header
        string sep = ";";
        reportFile << "Detector" << sep << "Descriptor" << sep << "Matcher" << sep
            << "DescriptorType" << sep << "Selector" << sep;

        std::vector<std::string> imgInfo = { "numKeypoint", "numKeypointsVehicle", "numKeypointsMatched", "tKeypointDetection", "tKeypointDesc" };

        stringstream ss;
        for (int i = 0; i < imgInfo.size(); ++i) {
            for (int j = 0; j < combinationInfo[0].numKeypoints.size(); ++j) {
                ss.clear();
                ss.str("");
                ss << imgInfo[i] << "_" << j << sep;
                reportFile << ss.str();
            }
        }

        // write results for each combination
        for (auto combination : combinationInfo) {
            ss.clear();
            ss.str("");

            ss << combination.detector << sep << combination.descriptor << sep << combination.matcherType << sep << combination.descriptorType << sep << combination.selectorType << sep;
            for (int j = 0; j < combination.numKeypoints.size(); ++j)
                ss << combination.numKeypoints[j] << sep;
            
            for (int j = 0; j < combination.numKeypointsVehicle.size(); ++j)
                ss << combination.numKeypointsVehicle[j] << sep;

            for (int j = 0; j < combination.numKeypointsMatched.size(); ++j)
                ss << combination.numKeypointsMatched[j] << sep;

            for (int j = 0; j < combination.tKeypointDetection.size(); ++j)
                ss << combination.tKeypointDetection[j] << sep;

            for (int j = 0; j < combination.tKeypointDescription.size(); ++j) 
                ss << combination.tKeypointDescription[j] << sep;

            ss << "\n";

            reportFile << ss.str();
        }
        reportFile << "\n";
        reportFile.close();
    }    

}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    // create all possible combinations
    std::vector<std::string> detectors = { "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT" };
    std::vector<std::string> descriptors = { "BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT" };
    std::vector<std::string> matcherType = { "MAT_BF", "MAT_FLANN" };
    std::vector<std::string> descriptorType = { "DES_BINARY", "DES_HOG" };
    std::vector<std::string> selectorType = { "SEL_NN", "SEL_KNN" };
    //std::vector<std::string> selectorType = { "SEL_KNN" };

    std::vector<DetectionInfo> combinationInfo;
    DetectionInfo info;
    for (auto det : detectors) {
        for (auto desc : descriptors) {
            for (auto mat : matcherType) {
                for (auto sel : selectorType) {
                    info.detector = det; 
                    info.descriptor = desc;
                    // descriptor type: SIFT -> HOG, else -> binary
                    if (desc.compare("SIFT") == 0)
                        info.descriptorType = descriptorType[1];
                    else
                        info.descriptorType = descriptorType[0];

                    info.matcherType = mat;
                    info.selectorType = sel;

                    if ((det.compare("SIFT") == 0) && (desc.compare("ORB") == 0)
                        || ((det.compare("AKAZE") == 0) && (desc.compare("AKAZE") != 0))
                        || ((det.compare("AKAZE") != 0) && (desc.compare("AKAZE") == 0)))
                        cout << "Invalid combination: detector " << det << "; descriptor " << desc << endl;
                    else 
                        combinationInfo.push_back(info);
                }
            }
        }
    }
    
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    //string dataPath = "../";
    string dataPath = "../../../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results    

    // outer loop over all possible combinations
     
    //for (auto combination : combinationInfo)
    for(std::vector<DetectionInfo>::iterator it = combinationInfo.begin(); it < combinationInfo.end(); ++it)
    {
        cout << "Detector: " << it->detector << endl;
        cout << "Descriptor: " << it->descriptor << endl;
        cout << "Desc. Type: " << it->descriptorType << endl;
        cout << "Matcher Type: " << it->matcherType << endl;
        cout << "Selector Type: " << it->selectorType << endl;
        
        // clear Buffer
        dataBuffer.clear();        

        /* MAIN LOOP OVER ALL IMAGES */

        for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
        {
            /* LOAD IMAGE INTO BUFFER */

            // assemble filenames for current index
            ostringstream imgNumber;
            imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
            string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

            // load image from file and convert to grayscale
            cv::Mat img, imgGray;
            img = cv::imread(imgFullFilename);
            cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

            //// STUDENT ASSIGNMENT
            //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize
            //// --> DONE

            // if dataBuffer has reached its maximum size before appending new image,
            // delete first element
            if (dataBuffer.size() == dataBufferSize)
                dataBuffer.erase(dataBuffer.begin());

            // push image into data frame buffer
            DataFrame frame;
            frame.cameraImg = imgGray;
            dataBuffer.push_back(frame);

            //// EOF STUDENT ASSIGNMENT
            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

            /* DETECT IMAGE KEYPOINTS */

            // extract 2D keypoints from current image
            vector<cv::KeyPoint> keypoints; // create empty feature list for current image
            //string detectorType = "SHITOMASI";
            //string detectorType = "SIFT";         // checked "SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"           
            //string detectorType = combination.detector;
            string detectorType = it->detector;

            //// STUDENT ASSIGNMENT
            //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
            //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
            //// --> DONE

            // detection time
            double t = 0.;
            if (detectorType.compare("SHITOMASI") == 0)
            {
                t = detKeypointsShiTomasi(keypoints, imgGray, false);
            }
            else if (detectorType.compare("HARRIS") == 0)
            {
                t = detKeypointsHarris(keypoints, imgGray, false);
            }
            else if ((detectorType.compare("FAST") == 0)
                | (detectorType.compare("BRISK") == 0)
                | (detectorType.compare("ORB") == 0)
                | (detectorType.compare("AKAZE") == 0)
                | (detectorType.compare("SIFT") == 0))
            {
                t = detKeypointsModern(keypoints, imgGray, detectorType, false);
            }
            else
            {
                cout << "Detectortype not recognized. Return." << endl;
                return -1;
            }

            // store information
            //combination.tKeypointDetection.push_back(t);
            it->tKeypointDetection.push_back(t);

            //// EOF STUDENT ASSIGNMENT

            // store information
            //combination.numKeypoints.push_back(static_cast<int>(keypoints.size()));
            it->numKeypoints.push_back(static_cast<int>(keypoints.size()));

            //// STUDENT ASSIGNMENT
            //// TASK MP.3 -> only keep keypoints on the preceding vehicle
            //// --> DONE

            // only keep keypoints on the preceding vehicle
            bool bFocusOnVehicle = true;
            // cv::Rect = (x,y, width, height)
            cv::Rect vehicleRect(535, 180, 180, 150);
            if (bFocusOnVehicle)
            {
                std::vector<cv::KeyPoint> keypointsRect;
                for (auto i : keypoints) {
                    if (vehicleRect.contains(i.pt))
                    {
                        keypointsRect.push_back(i);
                    }
                }
                keypoints = keypointsRect;
            }
            // store information
            //combination.numKeypointsVehicle.push_back(static_cast<int>(keypoints.size()));
            it->numKeypointsVehicle.push_back(static_cast<int>(keypoints.size()));

            //// EOF STUDENT ASSIGNMENT

            // optional : limit number of keypoints (helpful for debugging and learning)
            bool bLimitKpts = false;
            if (bLimitKpts)
            {
                int maxKeypoints = 50;

                if (detectorType.compare("SHITOMASI") == 0)
                { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                    keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                }
                cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                cout << " NOTE: Keypoints have been limited!" << endl;
            }

            // push keypoints and descriptor for current frame to end of data buffer
            (dataBuffer.end() - 1)->keypoints = keypoints;
            cout << "#2 : DETECT KEYPOINTS done" << endl;

            /* EXTRACT KEYPOINT DESCRIPTORS */

            //// STUDENT ASSIGNMENT
            //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
            //// -> BRIEF, ORB, FREAK, AKAZE, SIFT
            //// --> DONE

            cv::Mat descriptors;
            //string descriptorType = "SIFT"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
            //string descriptorType = combination.descriptor;
            string descriptorType = it->descriptor;
            t = 0.;
            t = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
            //// EOF STUDENT ASSIGNMENT

            // store information
            //combination.tKeypointMatching.push_back(t);
            it->tKeypointDescription.push_back(t);

            // push descriptors for current frame to end of data buffer
            (dataBuffer.end() - 1)->descriptors = descriptors;

            cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

            if (dataBuffer.size() > 1) // wait until at least two images have been processed
            {

                /* MATCH KEYPOINT DESCRIPTORS */

                vector<cv::DMatch> matches;
                //string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
                //string descriptorType = "DES_HOG";      // DES_BINARY, DES_HOG
                //string selectorType = "SEL_NN";         // SEL_NN, SEL_KNN

                //string matcherType = combination.matcherType;
                //string descriptorType = combination.descriptorType;
                //string selectorType = combination.selectorType;
                string matcherType = it->matcherType;
                string descriptorType = it->descriptorType;
                string selectorType = it->selectorType;

                //// STUDENT ASSIGNMENT
                //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                //// --> DONE
                //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
                //// --> DONE

                matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                    matches, descriptorType, matcherType, selectorType);

                //// EOF STUDENT ASSIGNMENT

                // store information
                //combination.numKeypointsMatched.push_back(static_cast<int>(matches.size()));
                it->numKeypointsMatched.push_back(static_cast<int>(matches.size()));

                // store matches in current data frame
                (dataBuffer.end() - 1)->kptMatches = matches;

                cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                // visualize matches between current and previous image
                bVis = false;
                if (bVis)
                {
                    cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                    cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                        matches, matchImg,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                    string windowName = "Matching keypoints between two camera images";
                    cv::namedWindow(windowName, 7);
                    cv::imshow(windowName, matchImg);
                    cout << "Press key to continue to next image" << endl;
                    cv::waitKey(0); // wait for key to be pressed
                }
                bVis = false;
            }

        } // eof loop over all images
    
    }       // eof loop over combinations

    saveReport(combinationInfo);

    return 0;
}
