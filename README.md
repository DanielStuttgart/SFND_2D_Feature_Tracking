# Sensor Fusion Nanodegree: Camera 2D Feature Tracking
Part of Sensor Fusion Nanodegree at Udacity

## Installation and Setup
For solving all given tasks, OpenCV 4.1 and its contributions (used for SIFT, BRIK - detectors) are needed. I installed OpenCV and OpenCV_Contrib via GIT and CMake (https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html). In Windows, the path to OpenCV needs to be added to system PATH-variable. Furthermore, an own OPENCV_DIR variable needs to be set. 

## Description of the Project Structure
* Segmentation
  * /src/quiz/ransac
* Clustering
  * /src/quiz/cluster
* Main project
  * /src
  
## Project Description

### MP.1 Use ring buffer
```c++
// if dataBuffer has reached its maximum size before appending new image,
// delete first element
if (dataBuffer.size() == dataBufferSize) 
    dataBuffer.erase(dataBuffer.begin());        

// push image into data frame buffer
DataFrame frame;
frame.cameraImg = imgGray;
dataBuffer.push_back(frame);
``` 

### MP.2 Add more keypoint detectors
See code for more information. 

### MP.3 Limit keypoint area to rectangle in front
```c++
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
```

### MP.4 Add more keypoint descriptors
See code for more information. Caution: It is not possible to randomly use each keypoint detector with each keypoint descriptor. (E.g. SIFT-keypoint detector does not work with ORB keypoint descriptor. SIFT- and ORB-keypoint detector do not work with AKAZE keypoint descriptor).

### MP.5 Add FLANN-matching
See code for more information. 

### MP.6 Add KNN match selection and perform descriptor distance ratio filtering
Instead of directly using the detected matches, a descriptor distance ratio filtering is done: 
```c++
// k nearest neighbors (k=2)
int k = 2;

// need to store in own knn_matches
std::vector< std::vector<cv::DMatch> > knn_matches;        
matcher->knnMatch(descSource, descRef, knn_matches, k);

// filter matches using the Lowe's ratio test
// taken from https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
const float ratio_thresh = 0.8f;        
for (size_t i = 0; i < knn_matches.size(); i++)
{
    if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    {
        matches.push_back(knn_matches[i][0]);
    }
}
```

### MP.7 Number of Keypoints for all images
The number of keypoints detected by each detector is recorded within a csv-file. Furthermore, the number of matched keypoints is compared to the number of detected keypoints within the preceding car's probable position. It can be seen, that the number of keypoints with HARRIS-detector are quite low, s.t. the quality of keypoint tracking may be influenced negatively.

### MP.8 Number of Keypoints with all possible combinations
The results for this task are stored within the csv-file SFND_FeatureTracking_Report.csv. Three numbers of keypoints are stored within this table: 
* total number of keypoints
* keypoints within preciding car region
* matched keypoints between two frames

### MP.9 Log the runtime of each detector and descriptor and choose best three combinations
The runtime stored within the csv-file SFND_FeatureTracking_Report.csv . For further analysis and better highlighting of relevant columns, the file is stored as Excel table SFND_Feature_Tracking_Report.xlsx within the root-folder. The FAST-detector is by factor 10 the fastest and should be considered for driver assistence / automated driving, since a fast reaction time of the car is needed in this area. Following combinations are the top 3 regarding the runtimes: 
No. | Detector | Descriptor
----|----------|---------
1 | FAST     | BRISK
2 | FAST     | BRIEF
3 | FAST     | ORB


