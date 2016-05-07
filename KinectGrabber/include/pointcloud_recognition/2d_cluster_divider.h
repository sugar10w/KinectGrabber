/*
 * Created by sugar10w, 2016.3.24
 * Last edited by sugar10w, 2016.5.7
 *
 * 提取二维轮廓, 将一个点云拆分为多个点云
 */

#ifndef __2D_CLUSTER_DIVIDER_H__
#define __2D_CLUSTER_DIVIDER_H__

#include <opencv2/opencv.hpp>

#include "common.h"
#include "pointcloud_recognition/object_cluster.h"

namespace tinker {
namespace vision {

class ClusterDivider2D
{
public:
    ClusterDivider2D(PointCloudConstPtr & cloud, cv::Mat & mask);
    void GetDividedCluster(std::vector<ObjectCluster>& divided_objects);
    inline cv::Mat GetMask() {return mask_;}
private:
    PointCloudConstPtr cloud_;
    cv::Mat mask_;
};

}   
}    

#endif // __2D_CLUSTER_DIVIDER_H__
