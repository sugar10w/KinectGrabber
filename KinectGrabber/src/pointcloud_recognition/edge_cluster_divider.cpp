/* 
 * Created by sugar10w, 2016.3.24
 * Last edited by sugar10w, 2016.3.24
 *
 * 利用shard图层直接获取Contours。
 * 之后参考no_plane图层确定所有Contours是否为平面。
 */


#include "pointcloud_recognition/edge_cluster_divider.h"
#include "pointcloud_filter/2dmask_filter.h"

namespace tinker {
namespace vision {


EdgeClusterDivider::EdgeClusterDivider(PointCloudConstPtr & cloud, cv::Mat & shard_mask, cv::Mat & no_plane_mask)
    : cloud_(cloud), mask_(cv::Mat::zeros(shard_mask.size(), CV_8UC1)),
    shard_mask_(shard_mask.clone()), no_plane_mask_(no_plane_mask)
{
}

std::vector<ObjectCluster> EdgeClusterDivider::GetDividedCluster()
{
    std::vector<ObjectCluster> divided_clouds;

    cv::Mat contour = ~shard_mask_;

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours( contour, contours, CV_RETR_CCOMP , CV_CHAIN_APPROX_NONE);

    for (int idx=0; idx<contours.size(); ++idx)
    {
        double cont_size = cv::contourArea(contours[idx]);
        if (cont_size>20000) continue;
        if (cont_size<100)
        {
            continue;
        }
        
        cv::Mat sub_mask(mask_.size(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(sub_mask, contours, idx, cv::Scalar(255), CV_FILLED, 4);
       
        double valid_area=0;
        for (int i=0; i<sub_mask.rows; ++i)
            for (int j=0; j<sub_mask.cols; ++j)
                if (sub_mask.at<uchar>(i,j) && no_plane_mask_.at<uchar>(i,j)) valid_area+=1;

        if (valid_area/cont_size>0.5)
        {
            ObjectCluster object_cluster( GetCloudFromMask(sub_mask, cloud_));
            if (object_cluster.isValid())
            {
                divided_clouds.push_back(object_cluster);
                mask_ |= sub_mask;
                 
                OpenImage(sub_mask, 6, 0, 0);
                shard_mask_ &= ~sub_mask;
            }
            else
            {
                OpenImage(sub_mask, 6, 0, 0);
                shard_mask_ &= ~sub_mask;
            }
        }
      /*  else
        {
            OpenImage(sub_mask, 6, 0, 0);
            shard_mask_ &= ~sub_mask;
        }*/
    }

    return divided_clouds;
}


}
}
    


