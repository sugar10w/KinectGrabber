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

void EdgeClusterDivider::GetDividedCluster(std::vector<ObjectCluster>& divided_objects)
{
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
        int min_x = 1000, min_y = 1000, max_x = 0, max_y = 0;
        for (int i=0; i<contours[idx].size(); ++i)
        {
            const cv::Point pt = contours[idx][i];
            if (min_x>pt.x) min_x=pt.x;
            if (min_y>pt.y) min_y=pt.y;
            if (max_x>pt.x) max_x=pt.x;
            if (max_y>pt.y) max_y=pt.y;
        }
        if (cont_size /((max_x-min_x)*(max_y*min_y)) < 0.4) continue;

        cv::Mat sub_mask(mask_.size(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(sub_mask, contours, idx, cv::Scalar(255), CV_FILLED, 4);
       
        cv::Mat plane_trial_mask = sub_mask.clone();
        OpenImage(plane_trial_mask, 0, 4, 0);
        int valid_area = 0, total_area = 0;
        for (int i=0; i<sub_mask.rows; ++i)
            for (int j=0; j<sub_mask.cols; ++j)
            {
                if (plane_trial_mask.at<uchar>(i,j)) 
                {
                    total_area+=1;
                    if (no_plane_mask_.at<uchar>(i,j)) valid_area+=1;
                }
            }

        if (total_area!=0 && (float)valid_area/(float)total_area>0.6)
        {
            OpenImage(sub_mask, 0, 1, 0);
            ObjectCluster object_cluster( GetCloudFromMask(sub_mask, cloud_));
            if (object_cluster.isValid())
            {
                divided_objects.push_back(object_cluster);
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
}


}
}
    


