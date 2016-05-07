/*
 * Created by sugar10w, 2016.3.24
 * Last edited by sugar10w, 2016.5.7
 *
 * 提取二维轮廓, 将一个点云拆分为多个点云
 */


#include "pointcloud_recognition/2d_cluster_divider.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pointcloud_filter/2dmask_filter.h"

namespace tinker {
namespace vision {


ClusterDivider2D::ClusterDivider2D(PointCloudConstPtr & cloud, cv::Mat & mask)
    : cloud_(cloud), mask_(mask.clone())
{
}

void ClusterDivider2D::GetDividedCluster(std::vector<ObjectCluster>& divided_objects)
{
    /* 获取mask_的边缘 */
    cv::Mat contour = ~mask_;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( contour, contours, CV_RETR_CCOMP , CV_CHAIN_APPROX_NONE);

    for (int idx=0; idx<contours.size(); ++idx)
    {
        /* 面积初判 */
        double cont_size = cv::contourArea(contours[idx]);
        if (cont_size>20000 || cont_size<80) continue;
         
        /* 尝试生成目标 */
        cv::Mat sub_mask(mask_.size(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(sub_mask, contours, idx, cv::Scalar(255), CV_FILLED, 4);
        OpenImage(sub_mask, 0, 1, 0);
        ObjectCluster object_cluster( GetCloudFromMask(sub_mask, cloud_));
        if (object_cluster.isValid())
            divided_objects.push_back(object_cluster);        
    }
}


}    
}    
