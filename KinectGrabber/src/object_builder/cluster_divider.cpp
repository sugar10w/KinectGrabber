/*
 * Created by 郭嘉丞, 2016.1
 * Last edited by sugar10w, 2016.2.29
 *
 * 将一个点云拆分为多个点云
 *
 */

#include "object_builder/cluster_divider.h"

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>


#include "common.h"
#include "pcl_filter/2dmask_filter.h"

namespace tinker
{
namespace vision
{

    ClusterDivider::ClusterDivider(PointCloudPtr point_cloud) 
        : point_cloud_(new PointCloud)
    {
        ////去除2m以外的点
        //pcl::PassThrough<PointT> pass;
        //pass.setInputCloud(point_cloud);
        //pass.setFilterFieldName("z");
        //pass.setFilterLimits(0.5, 2.00);
        //pass.filter(*point_cloud_);
        point_cloud_ = point_cloud;

        //确定降采样叶子的大小
        float min_x=1e9, max_x=-1e9;
        for (int i=0; i<point_cloud_->width*point_cloud_->height; ++i)
        {
            PointT & point = point_cloud_->points[i];
            if (point.x<min_x) min_x = point.x;
            if (point.x>max_x) max_x = point.x;
        }
        if (min_x < max_x)
        {
            leaf_size_=(max_x - min_x) / 100;
            if (leaf_size_<0.03f) leaf_size_=0.03f;
        }
        else leaf_size_ = 0.03f;
    }

    std::vector<ObjectCluster> ClusterDivider::GetDividedCluster()
    {
        std::vector<ObjectCluster> divided_clouds;
        if (point_cloud_->points.size()==0) return divided_clouds;
       
        //先降采样
        PointCloudPtr cloud_filtered (new PointCloud);        
        pcl::VoxelGrid<PointT> vg;
        vg.setInputCloud (point_cloud_);
        vg.setLeafSize (leaf_size_, leaf_size_, leaf_size_);  // TODO set it manually
        vg.filter (*cloud_filtered);

        //KdTree
        pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
        tree->setInputCloud (cloud_filtered);

        //欧几里得聚类
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance (leaf_size_); // 4cm
        ec.setMinClusterSize (10);
        //ec.setMaxClusterSize (8000);
        ec.setSearchMethod (tree);
        ec.setInputCloud (cloud_filtered);
        ec.extract (cluster_indices);

        //逐个点云处理
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {
            PointCloudPtr cloud_cluster(new PointCloud);
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
                cloud_cluster->points.push_back (cloud_filtered->points[*pit]);
            cloud_cluster->width = cloud_cluster->points.size ();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;

            ObjectCluster object_cluster(cloud_cluster, leaf_size_);
            divided_clouds.push_back(object_cluster);
        }
        return divided_clouds;
    } 


}
}

