/*
 * Created by sugar10w, 2016.3.18
 * Last edited by sugar10w, 2016.3.18
 * 
 * 基于从KinectSDK2获取的数据。
 * 获取512*424的二维蒙版，应用到原始点云上。
 *
 */

#ifndef _POINTCLOUD_2DMASK_FILTER__
#define _POINTCLOUD_2DMASK_FILTER__

#include "common.h"
#include <opencv2/opencv.hpp>

namespace tinker {
namespace vision {

/*获取512*424的二维蒙版，应用到原始点云上*/
PointCloudPtr GetCloudFromMask(const cv::Mat & mask, const PointCloudConstPtr & cloud);    

/*获取有效点蒙版*/
cv::Mat GetValidMask(PointCloudConstPtr cloud);

/* 二维操作 */
void OpenImage(cv::Mat & img, 
     int refill_kernel_size,
     int erode_kernel_size,
     int dilate_kernel_size);

/* 对三维的点云进行二维操作 */
void OpenImage(PointCloudPtr & cloud, 
     int refill_kernel_size,
     int erode_kernel_size,
     int dilate_kernel_size);

/* 处理Kinect在物体边缘部分产生的拉伸散点 */
cv::Mat GetShardMask(PointCloudConstPtr & cloud);

}    
}

#endif //_POINTCLOUD_2DMASK_FILTER__
