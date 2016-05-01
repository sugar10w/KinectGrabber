/*
 * Created by sugar10w, 2016.2.27 
 * Last edited by sugar10w, 2016.2.28
 *
 * 找到点云中的平面
 * 从其他点云中去除平面
 *
 */

#include"pcl_filter/plane_filter.h"

#include<iostream>
#include<vector>

#include<pcl/sample_consensus/method_types.h>
#include<pcl/sample_consensus/model_types.h>
#include<pcl/sample_consensus/sac_model_plane.h>  
#include<pcl/segmentation/sac_segmentation.h>
#include<pcl/filters/voxel_grid.h>
#include<pcl/filters/extract_indices.h>

namespace tinker {
namespace vision {    

PlaneFilter::PlaneFilter(PointCloudPtr &cloud, float leaf_size)
    : leaf_size_(leaf_size)
{
    /*降采样*/
    PointCloudPtr downsampled_cloud(new PointCloud);
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    sor.filter(*downsampled_cloud);

    /* prepare */
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    /*segmentation*/
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);
    seg.setInputCloud(downsampled_cloud);
    seg.segment(*inliers, *coefficients);

    if (coefficients->values.size()==4)
    {
        /*coefficients to plane_   TODO 直接采用统一格式*/
        //plane_.resize(4);
        //plane_(0) = coefficients->values[0];
        //plane_(1) = coefficients->values[1];
        //plane_(2) = coefficients->values[2];
        //plane_(3) = coefficients->values[3];
        planes_.push_back(*coefficients);
        plane_found_ = true;
    }
    else
    {
        plane_found_ = false;
    }

}

PlaneFilter::PlaneFilter(std::vector<pcl::ModelCoefficients> planes)
{
    planes_ = planes;
    plane_found_ = planes_.size()!=0;
}

PlaneFilter::~PlaneFilter()
{
}

void PlaneFilter::Filter(PointCloudPtr &cloud, float remove_radius)
{
    if (!plane_found_) return;

    for (int i=0; i<planes_.size(); ++i)
    {
        Eigen::VectorXf plane_; plane_.resize(4);
        plane_(0) = planes_[i].values[0];
        plane_(1) = planes_[i].values[1];
        plane_(2) = planes_[i].values[2];
        plane_(3) = planes_[i].values[3];

        /* 取出点 */
        pcl::IndicesPtr inliers(new std::vector<int>);
        pcl::SampleConsensusModelPlane<PointT>::Ptr plane_model(
                new pcl::SampleConsensusModelPlane<PointT>(cloud));
        plane_model->selectWithinDistance(plane_, remove_radius, *inliers);

        //std::cout<<plane_<<std::endl;
        //std::cout<<"remove "<<inliers->size()<<" points"<<std::endl;

        /* extract*/
        /*PointCloudPtr cloud_no_plane(new PointCloud);
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_no_plane);
        cloud = cloud_no_plane;*/
        
        /* Softly disable */
        for (int i=0; i<inliers->size(); ++i)
        {
            int t = (*inliers)[i];
            PointT & pt = cloud->points[t];
            pt.x = pt.y = pt.z = 0;
        }

    }
}

/* 获取过滤蒙版 */
cv::Mat PlaneFilter::GetMask(PointCloudConstPtr &cloud, float remove_radius)
{
    cv::Mat mask = cv::Mat::zeros(cloud->height, cloud->width, CV_8UC1);
    if (!plane_found_) return ~mask;

    for (int i=0; i<planes_.size(); ++i)
    {
        Eigen::VectorXf plane_; plane_.resize(4);
        plane_(0) = planes_[i].values[0];
        plane_(1) = planes_[i].values[1];
        plane_(2) = planes_[i].values[2];
        plane_(3) = planes_[i].values[3];

        /* 取出点 */
        pcl::IndicesPtr inliers(new std::vector<int>);
        pcl::SampleConsensusModelPlane<PointT>::Ptr plane_model(
                new pcl::SampleConsensusModelPlane<PointT>(cloud));
        plane_model->selectWithinDistance(plane_, remove_radius, *inliers);
        
        /* mask */
        for (int i=0; i<inliers->size(); ++i)
        {
            int t = (*inliers)[i];
            int y = t / cloud->width, x = cloud->width - 1 - t%cloud->width;
            mask.at<uchar>(y, x) = 255;
        }
    }
    return ~mask;
}

bool PlaneFilter::PlaneFound()
{
    return plane_found_;
}

}
}
