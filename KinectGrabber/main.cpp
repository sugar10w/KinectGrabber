/*
 * Created by UnaNancyOwen, 2015.11 (https://github.com/UnaNancyOwen/KinectGrabber/)
 * Last edited by sugar10w, 2016.5.1
 * 
 * tinker2016的子项目. 
 * 显示在三维场景中, 尤其是桌面或者橱柜场景中, 找到的物体.
 *
 */

// Disable Error C4996 that occur when using Boost.Signals2.
#ifdef _DEBUG
#define _SCL_SECURE_NO_WARNINGS
#endif

#include "kinect2_grabber.h"

#include <boost/make_shared.hpp>

#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/opencv.hpp>

#include "pointcloud_filter/2dmask_filter.h"
#include "pointcloud_filter/rgbd_plane_detector.h"
#include "pointcloud_recognition/2d_cluster_divider.h"
#include "pointcloud_recognition/edge_cluster_divider.h"
#include "common.h"

using namespace tinker::vision;

pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer( "3D Object Detector" ) );
bool flag_paused = false;   //p
bool flag_draw_box = false; //d
bool flag_debug = false;    //b
bool flag_color = false;    //v

/* 设置键盘响应 */
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event)
{
    if (event.getKeySym() == "p" && event.keyDown())
    {
        flag_paused = !flag_paused;
    }
    if (event.getKeySym() == "d" && event.keyDown())
    {
        flag_draw_box = !flag_draw_box;
    }
    if (event.getKeySym() == "b" && event.keyDown())
    {
        flag_debug = !flag_debug;
    }
    if (event.getKeySym() == "v" && event.keyDown())
    {
        flag_color = !flag_color;
    }
    if (event.getKeySym() == "r" && event.keyDown())
    {
        viewer->setCameraPosition( 0.0, 0.0, -2.5, 0.0, 0.0, 0.0 );
    }
}

int main( int argc, char* argv[] )
{
    /* 设置viewer基本参数 */
    viewer->setCameraPosition( 0.0, 0.0, -2.5, 0.0, 0.0, 0.0 );
    viewer->setBackgroundColor(0.8, 0.8, 0.8);
    /* 关联键盘响应 */
    viewer->registerKeyboardCallback(keyboardEventOccurred);

    /* ConstPtr点云 */
    pcl::PointCloud<PointT>::ConstPtr cloud;

    /* 设置进程锁(MUTual EXclusion); 设置从Grabber获取点云的函数; */
    boost::mutex mutex;
    boost::function<void( const pcl::PointCloud<PointT>::ConstPtr& )> function =
        [&cloud, &mutex]( const pcl::PointCloud<PointT>::ConstPtr& ptr ){
            boost::mutex::scoped_lock lock( mutex );
            cloud = ptr;
        };

    /* 设置Kinect2Grabber; 关联获取点云函数; 开始连接Kinect2; */
    boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::Kinect2Grabber>();
    boost::signals2::connection connection = grabber->registerCallback( function );
    grabber->start();

    while( !viewer->wasStopped() )
    {
        /* 更新viewer */
        viewer->spinOnce();

        /* 检查进程锁 */
        boost::mutex::scoped_try_lock lock( mutex );
        if( cloud && lock.owns_lock() )
        {
            if( cloud->size() != 0 )
            {
                if (flag_paused)
                {
                    viewer->spinOnce(500);
                    continue;
                }

                /* 处理点云 */

                /* 去平面 */
                cv::Mat no_plane_mask = GetNoPlaneMask(cloud);
                OpenImage(no_plane_mask, 0, 2, 0);
                if (flag_debug) cv::imshow("no_plane_mask", no_plane_mask);

                /* 边缘拉伸点 */
                cv::Mat shard_mask = ~GetShardMask(cloud);
                //OpenImage(shard_mask, 0, 1, 0);
                if (flag_debug) cv::imshow("shard_mask", shard_mask);

                /* EdgeClusterDivider */
                EdgeClusterDivider cluster_divider_0(cloud, shard_mask, no_plane_mask);
                std::vector<ObjectCluster> clusters_0 = cluster_divider_0.GetDividedCluster();
                /* 更新shard_mask, 防止区域的重复判定 */
                shard_mask = cluster_divider_0.GetShardMask();

                /* (待判定)ClusterDivider2D */
                OpenImage(no_plane_mask, 0, 2, 0);
                cv::Mat mask = shard_mask & no_plane_mask;
                ClusterDivider2D cluster_divider_1(cloud, mask);
                std::vector<ObjectCluster> clusters_1;
                /* 判定是否进行第2步 */
                if (clusters_0.size() <= 5)
                {
                    clusters_1 = cluster_divider_1.GetDividedCluster();
                    /* 综合两种提取方式的结果 */
                    mask = cluster_divider_0.GetMask() | cluster_divider_1.GetMask();
                }
                else 
                {
                    mask = cluster_divider_0.GetMask();
                }

                if (flag_color)
                {
                    /* 获取二维BGR图片 */
                    cv::Mat img(cloud->height, cloud->width, CV_8UC3);
                    const PointT* pt = & cloud->points[0];
                    for (int i=0; i<cloud->height; ++i)
                        for (int j=cloud->width-1; j>=0; --j)
                        {
                            cv::Vec3b & img_point = img.at<cv::Vec3b>(i, j);
                            img_point[0] = pt->b;
                            img_point[1] = pt->g;
                            img_point[2] = pt->r;
                            ++pt;
                        }
                    cv::imshow("raw RGB", img);

                    /* 显示二维过滤结果 */
                    for (int i=0; i<mask.rows; ++i)
                        for (int j=0; j<mask.cols; ++j)
                            if (! mask.at<uchar>(i,j))
                            {
                                cv::Vec3b & point = img.at<cv::Vec3b>(i,j);
                                /* 背景部分 灰度, 变暗 */
                                point[0] = point[1] = point[2] = 
                                    (point[0]+point[1]+point[2])/6;
                            }
                    cv::imshow("2D Result", img);
                }

                /* 显示点云 */
                PointCloudPtr display_cloud = GetCloudFromMask(mask, cloud);
                if( !viewer->updatePointCloud( display_cloud, "cloud" ) )
                    viewer->addPointCloud(display_cloud, "cloud");
                
                if (flag_draw_box)
                {
                    viewer->removeAllShapes();
                    for (int i=0; i<clusters_0.size(); ++i)
                    {
                        clusters_0[i].DrawBoundingBox(*viewer, i);
                    }
                    for (int i=0;  i<clusters_1.size(); ++i)
                    {
                        clusters_1[i].DrawBoundingBox(*viewer, i);
                    }
                    flag_draw_box = false;
                }
            }
            else
            {
                std::cout<<" Warning: No cloud received!"<<std::endl;
            }
        }
    }

    // Stop Grabber
    grabber->stop();

    return 0;
}