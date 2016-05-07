/* 
 * Created by sugar10w, 2016.3.24
 * Last edited by sugar10w, 2016.5.7
 *
 * 利用shard图层获取轮廓.
 * 之后参考no_plane图层, 将非平面的轮廓视为目标.
 */


#include "pointcloud_recognition/edge_cluster_divider.h"
#include "pointcloud_filter/2dmask_filter.h"

namespace tinker {
namespace vision {


EdgeClusterDivider::EdgeClusterDivider(PointCloudConstPtr & cloud, cv::Mat & shard_mask, cv::Mat & no_plane_mask)
    : cloud_(cloud), mask_(cv::Mat::zeros(shard_mask.size(), CV_8UC1)),
    shard_mask_(shard_mask), no_plane_mask_(no_plane_mask)
{
}

void EdgeClusterDivider::GetDividedCluster(std::vector<ObjectCluster>& divided_objects)
{
    /* 从shard_mask上获取轮廓 */
    cv::Mat contour = ~shard_mask_;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( contour, contours, CV_RETR_CCOMP , CV_CHAIN_APPROX_NONE);

    for (int idx=0; idx<contours.size(); ++idx)
    {
        /* 面积初判 */
        double cont_size = cv::contourArea(contours[idx]);
        if (cont_size>20000 || cont_size<80) continue;        
        
        /* sub_mask目标所在蒙版 */
        cv::Mat sub_mask(mask_.size(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(sub_mask, contours, idx, cv::Scalar(255), CV_FILLED, 4);
        
        /* 收缩4之后, 统计内部平面率 */
        cv::Mat plane_trial_mask = sub_mask.clone();
        OpenImage(plane_trial_mask, 0, 4, 0);
        int valid_area = 0, total_area = 0;
        for (int i=0; i<sub_mask.rows; ++i)
            for (int j=0; j<sub_mask.cols; ++j)
                if (plane_trial_mask.at<uchar>(i,j))
                {
                    total_area+=1;
                    if (no_plane_mask_.at<uchar>(i,j)) valid_area+=1;
                }
        /* 检查平面率 */
        if (total_area==0 || (float)valid_area/(float)total_area>0.6)
        {
            /* 尝试生成目标 */
            OpenImage(sub_mask, 0, 1, 0);
            ObjectCluster object_cluster( GetCloudFromMask(sub_mask, cloud_));
            if (object_cluster.isValid())
            {
                /* 在mask_上标记结果, 方便点云的展示 */
                divided_objects.push_back(object_cluster);
                mask_ |= sub_mask;
            }
        }
    }
}


}
}
    


