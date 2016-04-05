// Disable Error C4996 that occur when using Boost.Signals2.
#ifdef _DEBUG
#define _SCL_SECURE_NO_WARNINGS
#endif

#include "kinect2_grabber.h"

#include <boost/make_shared.hpp>

#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/opencv.hpp>

#include "pcl_filter/2dmask_filter.h"
#include "pcl_filter/depth_edge_detector.h"
#include "object_builder/2d_cluster_divider.h"
#include "object_builder/edge_cluster_divider.h"
#include "common.h"

using namespace tinker::vision;

bool flag_paused = false;
bool flag_draw_box = false;


void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event)
{
    if (event.getKeySym () == "p" && event.keyDown ())
    {
        flag_paused = !flag_paused;
    }
    if (event.getKeySym () == "d" && event.keyDown ())
    {
        flag_draw_box = !flag_draw_box;
    }
}

int main( int argc, char* argv[] )
{
    // PCL Visualizer
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
        new pcl::visualization::PCLVisualizer( "3D Object Detector" ) );
    viewer->setCameraPosition( 0.0, 0.0, -2.5, 0.0, 0.0, 0.0 );
    viewer->setBackgroundColor(0.8, 0.8, 0.8);
    
    viewer->registerKeyboardCallback(keyboardEventOccurred);

    // Point Cloud
    pcl::PointCloud<PointT>::ConstPtr cloud;

    // Retrieved Point Cloud Callback Function
    boost::mutex mutex;
    boost::function<void( const pcl::PointCloud<PointT>::ConstPtr& )> function =
        [&cloud, &mutex]( const pcl::PointCloud<PointT>::ConstPtr& ptr ){
            boost::mutex::scoped_lock lock( mutex );
            cloud = ptr;
        };

    // Kinect2Grabber
    boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::Kinect2Grabber>();

    // Register Callback Function
    boost::signals2::connection connection = grabber->registerCallback( function );

    // Start Grabber
    grabber->start();

    while( !viewer->wasStopped() ){
        // Update Viewer
        viewer->spinOnce();

        boost::mutex::scoped_try_lock lock( mutex );
        if( cloud && lock.owns_lock() ){
            if( cloud->size() != 0 ){

                if (flag_paused)
                {
                    viewer->spinOnce(500);
                    continue;
                }

                /* Processing to Point Cloud */

                //获取二维图片
                cv::Mat img(cloud->height, cloud->width, CV_8UC3);
                int k = 0;
                for (int j=0; j<cloud->height; ++j)
                    for (int i=cloud->width-1; i>=0; --i)
                    {
                        cv::Vec3b & img_point = img.at<cv::Vec3b>(j, i);
                        const PointT & pt = cloud->points[k];
                        img_point[0] = pt.b;
                        img_point[1] = pt.g;
                        img_point[2] = pt.r;
                        k++;
                    }
                cv::imshow("raw RGB", img);

                /* 去平面 */
                cv::Mat no_plane_mask = GetNoPlaneMask(cloud);
                OpenImage(no_plane_mask, 0, 2, 0);
                cv::imshow("no_plane_mask", no_plane_mask);

                /* 边缘拉伸点 */
                cv::Mat shard_mask = ~GetShardMask(cloud);
                //OpenImage(shard_mask, 0, 1, 0);
                cv::imshow("shard_mask", shard_mask);

                /* EdgeClusterDivider */
                EdgeClusterDivider pre_cluster_divider(cloud, shard_mask, no_plane_mask);
                std::vector<ObjectCluster> pre_clusters = pre_cluster_divider.GetDividedCluster();
                shard_mask = pre_cluster_divider.GetShardMask();

                /* ClusterDivider2D */
                OpenImage(no_plane_mask, 0, 2, 0);
                cv::Mat mask = shard_mask & no_plane_mask;
                ClusterDivider2D cluster_divider(cloud, mask);
                std::vector<ObjectCluster> clusters;
                
                /* 判定是否进行第2步 */
                if (pre_clusters.size() <= 5)
                {
                    clusters = cluster_divider.GetDividedCluster();
                    /* 综合两种提取方式的结果 */
                    mask = pre_cluster_divider.GetMask() | cluster_divider.GetMask();
                }
                else 
                {
                    mask = pre_cluster_divider.GetMask();
                }

                /* 显示二维过滤结果 */
                for (int i=0; i<mask.rows; ++i)
                    for (int j=0; j<mask.cols; ++j)
                        if (! mask.at<uchar>(i,j))
                        {
                            cv::Vec3b & point = img.at<cv::Vec3b>(i,j);
                            point[0] = point[1] = point[2] = 
                                (point[0]+point[1]+point[2])/6;
                            /*point[0] >>= 2;
                            point[1] >>= 2;
                            point[2] >>= 2;*/
                        }
                cv::imshow("2D Result", img);

                /* 显示点云 */
                PointCloudPtr display_cloud = GetCloudFromMask(mask, cloud);
                if( !viewer->updatePointCloud( display_cloud, "cloud" ) )
                    viewer->addPointCloud(display_cloud, "cloud");
                
                if (flag_draw_box)
                {
                    viewer->removeAllShapes();
                    for (int i=0; i<clusters.size(); ++i)
                    {
                        clusters[i].DrawBoundingBox(*viewer, i);
                    }
                    for (int i=0;  i<pre_clusters.size(); ++i)
                    {
                        pre_clusters[i].DrawBoundingBox(*viewer, i);
                    }
                    flag_draw_box = false;
                }
            }
        }
    }

    // Stop Grabber
    grabber->stop();

    return 0;
}