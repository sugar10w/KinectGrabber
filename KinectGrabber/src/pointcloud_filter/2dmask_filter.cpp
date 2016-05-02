/*
 * Created by sugar10w, 2016.3.18
 * Last edited by sugar10w, 2016.3.18
 * 
 * 基于从KinectSDK2获取的数据。
 * 获取512*424的二维蒙版，应用到原始点云上。
 *
 */

#include "pointcloud_filter/2dmask_filter.h"

namespace tinker {
namespace vision {

PointCloudPtr GetCloudFromMask(const cv::Mat & mask, const PointCloudConstPtr & cloud)
{
    assert(mask.type() == CV_8UC1);
    assert(cloud->height == mask.rows);
    assert(cloud->width == mask.cols);
    
    std::vector<int> indices;
    
    //注意这里的统一性
    int k = 0;
    for (int i=0; i<cloud->height; ++i)
        for (int j=cloud->width-1; j>=0; --j)
        {
            uchar img_point = mask.at<uchar>(i, j);
            if (img_point) indices.push_back(k);
            ++k;
        }

    PointCloudPtr extracted_cloud(new PointCloud);
    extracted_cloud->width = indices.size();
    extracted_cloud->height = 1;
    extracted_cloud->is_dense = true;

    for (int i=0; i<indices.size(); ++i)
    {
        extracted_cloud->points.push_back(
            cloud->points[ indices[i] ]);
    }

    return extracted_cloud;
}

cv::Mat GetValidMask(PointCloudConstPtr cloud)
{
    cv::Mat mask(cloud->height, cloud->width, CV_8UC1);

    int k = 0;
    for (int i=0; i<cloud->height; ++i)
        for (int j=cloud->width-1; j>=0; --j)
        {
            const PointT & pt = cloud->points[k];
            if (pt.z>=0.8 && pt.z<=10 && pt.r+pt.g+pt.b!=0)
                mask.at<uchar>(i, j) = 255;
            else
                mask.at<uchar>(i, j) = 0;
            k++;
        }
    return mask;
}

void OpenImage(cv::Mat & img, 
     int refill_kernel_size,
     int erode_kernel_size,
     int dilate_kernel_size)
{
    if (refill_kernel_size > 0)
     {
         cv::Mat refill_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
             cv::Size(2 * refill_kernel_size + 1, 2 * refill_kernel_size + 1),
             cv::Point(refill_kernel_size, refill_kernel_size));
         cv::dilate(img, img, refill_kernel);
     }
 
     if (erode_kernel_size > 0)
     {
         cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
             cv::Size(2 * erode_kernel_size + 1, 2 * erode_kernel_size + 1),
             cv::Point(erode_kernel_size, erode_kernel_size));
         cv::erode(img, img, erode_kernel);
     }
 
     if (dilate_kernel_size > 0)
     {
         cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
             cv::Size(2 * dilate_kernel_size + 1, 2 * dilate_kernel_size + 1),
             cv::Point(dilate_kernel_size, dilate_kernel_size));
         cv::dilate(img, img, dilate_kernel);
     }
}

void OpenImage(PointCloudPtr & cloud, 
     int refill_kernel_size,
     int erode_kernel_size,
     int dilate_kernel_size)
{
    cv::Mat img = GetValidMask(cloud);

    OpenImage(img, refill_kernel_size, erode_kernel_size, dilate_kernel_size);

    cloud = GetCloudFromMask(img, cloud);
}

/* 处理Kinect在物体边缘部分产生的拉伸散点 */
cv::Mat GetShardMask(PointCloudConstPtr & cloud)
{
    const float dist_limit = 0.01*0.01;

    cv::Mat mask = cv::Mat::zeros(cloud->height, cloud->width, CV_8UC1);

    // 注意统一这里的格式
    int k = 0;
    for (int i=0; i<cloud->height; ++i)
        for (int j=cloud->width-1; j>=0; --j)
        {
            if (cloud->points[k].z>2.00 || cloud->points[k].z<0.5) 
            {
                mask.at<uchar>(i,j)=255;
                k++;
                continue;
            }

            if (j==0 || j+1==cloud->width 
                || i==0 || i+1==cloud->height)
            {
                k++;
                continue;
            }
            const PointT pt0 = cloud->points[k],
                         pt1 = cloud->points[k+1],
                         pt2 = cloud->points[k+cloud->width];

            pcl::PointXYZ pt_1 (
                (pt0.x-pt1.x)*(pt0.x-pt1.x),
                (pt0.y-pt1.y)*(pt0.y-pt1.y),
                (pt0.z-pt1.z)*(pt0.z-pt1.z)),
                        pt_2 (
                (pt0.x-pt2.x)*(pt0.x-pt2.x),
                (pt0.y-pt2.y)*(pt0.y-pt2.y),
                (pt0.z-pt2.z)*(pt0.z-pt2.z));
            if ( pt_1.x+pt_1.y+pt_1.z>dist_limit
              || pt_2.x+pt_2.y+pt_2.z>dist_limit
              || pt_1.z > (pt_1.x+pt_1.y)*9
              || pt_2.z > (pt_2.x+pt_2.y)*9
             )
                mask.at<uchar>(i,j)=255;
            k++;
        }
    return mask;
}

}
}