/*
 * Created by sugar10w, 2016.3.19
 *
 * 在一张记录深度信息的cv::Mat上查找边缘。
 */

#ifndef _KINECT_DEPTH_CONTOUR__
#define _KINECT_DEPTH_CONTOUR__

#include <opencv2/opencv.hpp>
#include "common.h"

namespace tinker {
namespace vision {


/* 用pcl::PointXYZRGBNomal查找边缘。 */
//cv::Mat GetEdgeFromDepth(PointCloudNTPtr & cloud);

/* 为了处理前端细小物体的干扰而导致区域被切分的情况，尝试对相邻的区域进行辨别形式的连接操作。 */
//void ConnectDepthMask(PointCloudConstPtr & cloud, cv::Mat & mask);

/* 从边缘图像上，过滤平面获取蒙版 */
//cv::Mat GetMaskFromEdge(PointCloudConstPtr & cloud, PointCloudNTPtr & normals, cv::Mat & edge);



/* 基于法向量去除点云中的平面 */
cv::Mat GetNoPlaneMask(PointCloudConstPtr & cloud);


}
}


#endif //_KINECT_DEPTH_CONTOUR__
