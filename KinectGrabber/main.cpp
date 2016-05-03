/*
 * Created by UnaNancyOwen, 2015.11 (https://github.com/UnaNancyOwen/KinectGrabber/)
 * Last edited by sugar10w, 2016.5.1
 * 
 * tinker2016������Ŀ. 
 * ��ʾ����ά������, ������������߳��񳡾���, �ҵ�������.
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



/* ��ʾ����, �Լ����̿��Ƶĸ������� */
pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer( "3D Object Detector" ) );
bool flag_all = false;      //a
bool flag_paused = false;   //p
bool flag_draw_box = false; //d
bool flag_debug = false;    //b
bool flag_color = false;    //v
bool flag_integral = false; //i
bool flag_lock = false;     //l

/* �ر��, ���ڻ���mask���趨 */
cv::Mat integral_mask(424, 512, CV_8UC1, cv::Scalar(0));
int count = 0;
const int MAX_COUNT = 200;

/* ��������еĸ����ؼ��ɰ� */
//cv::Mat lock_mask(424, 512, CV_8UC1, cv::Scalar(0));
cv::Mat mask(424, 512, CV_8UC1, cv::Scalar(0));
cv::Mat shard_mask(424, 512, CV_8UC1, cv::Scalar(0));
cv::Mat no_plane_mask(424, 512, CV_8UC1, cv::Scalar(0));
const cv::Mat mask_all(424, 512, CV_8UC1, cv::Scalar(255));
/* BGRͼ�� */
cv::Mat img(424, 512, CV_8UC3, cv::Scalar(0,0,0));
cv::Mat img_result(424, 512, CV_8UC3, cv::Scalar(0,0,0));
/* Ŀ��vector */
std::vector<ObjectCluster> divided_objects;

/* ������Ӧ */
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event)
{
    if (event.getKeySym() == "a" && event.keyDown())
    {
        flag_all = !flag_all;
    }
    if (event.getKeySym() == "p" && event.keyDown())
    {
        flag_paused = !flag_paused;

        if (flag_paused)
        {
            flag_lock = true;
            std::cout<<"��ֹͣ���յ�����Ϣ. ����������ɰ�. ����ͣ���ɰ�ļ���."<<std::endl;
        }
        else
        {
            flag_lock = false;
            std::cout<<"��ʼ���յ�����Ϣ. ��ȡ���Խ���ɰ������."<<std::endl;
        }
    }
    if (event.getKeySym() == "d" && event.keyDown())
    {
        flag_draw_box = !flag_draw_box;
        if (flag_draw_box)
        {
            viewer->removeAllShapes();
            for (int i=0; i<divided_objects.size(); ++i)
                divided_objects[i].DrawBoundingBox(*viewer, i);
            std::cout<<"�ҵ�"<<divided_objects.size()<<"��Ŀ��."<<std::endl;
        }
        else
        {
            viewer->removeAllShapes();
        }
    }
    if (event.getKeySym() == "b" && event.keyDown())
    {
        flag_debug = !flag_debug;
        if (flag_debug)
        {
            cv::imshow("no_plane_mask", no_plane_mask);
            cv::imshow("shard_mask", shard_mask);
        }
    }
    if (event.getKeySym() == "v" && event.keyDown())
    {
        flag_color = !flag_color;
        if (flag_color)
        {
            cv::imshow("raw RGB", img);
            cv::imshow("2D Result", img_result);
        }
    }
    if (event.getKeySym() == "i" && event.keyDown())
    {
        flag_integral = !flag_integral;

        if (flag_integral)
        {
            integral_mask = 0;
            std::cout<<"�ѽ������ģʽ, ��ʼ�ɼ�"<<MAX_COUNT<<"֡ͼ��."<<std::endl;
        }
        else
        {
            count = 0;
            std::cout<<"���˳�����ģʽ. "<<std::endl;
        }
    }
    if (event.getKeySym() == "l" && event.keyDown())
    {
        if (flag_paused)
        {
            std::cout<<"�����˳�p״̬(��ͣ���յ���), �ٳ�����������ɰ�! "<<std::endl;
        }
        else
        {
            flag_lock = !flag_lock;
        
            if (flag_lock)
                std::cout<<"����������ɰ�. ����ͣ���ɰ�ļ���."<<std::endl;
            else
                std::cout<<"��ȡ���Խ���ɰ������. "<<std::endl;
        }
    }

    if (event.getKeySym() == "r" && event.keyDown())
    {
        viewer->setCameraPosition( 0.0, 0.0, -2.5, 0.0, 0.0, 0.0 );
        std::cout<<"�ӽ�������."<<std::endl;
    }
}

int main( int argc, char* argv[] )
{
    /* ����viewer�������� */
    viewer->setCameraPosition( 0.0, 0.0, -2.5, 0.0, 0.0, 0.0 );
    viewer->setBackgroundColor(0.8, 0.8, 0.8);
    /* ����������Ӧ */
    viewer->registerKeyboardCallback(keyboardEventOccurred);

    /* ConstPtr���� */
    pcl::PointCloud<PointT>::ConstPtr cloud;


    /* ���ý�����(MUTual EXclusion); ���ô�Grabber��ȡ���Ƶĺ���; */
    boost::mutex mutex;
    boost::function<void( const pcl::PointCloud<PointT>::ConstPtr& )> function =
        [&cloud, &mutex]( const pcl::PointCloud<PointT>::ConstPtr& ptr ){
            boost::mutex::scoped_lock lock( mutex );
            cloud = ptr;
        };

    /* ����Kinect2Grabber; ������ȡ���ƺ���; ��ʼ����Kinect2; */
    boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::Kinect2Grabber>();
    boost::signals2::connection connection = grabber->registerCallback( function );
    grabber->start();

    while( !viewer->wasStopped() )
    {
        /* ����viewer */
        viewer->spinOnce();

        /* �������� */
        boost::mutex::scoped_try_lock lock( mutex );
        if( cloud && lock.owns_lock() )
        {
            if( cloud->size() != 0 )
            {
                if (flag_paused)
                {
                    viewer->spinOnce(200);
                    continue;
                }

                if (!flag_lock)
                {
                    /* ������� */
                    divided_objects.clear();

                    /* ȥƽ�� */
                    no_plane_mask = GetNoPlaneMask(cloud);
                    OpenImage(no_plane_mask, 0, 2, 0);
                    if (flag_debug) cv::imshow("no_plane_mask", no_plane_mask);

                    /* ��Ե����� */
                    shard_mask = ~GetShardMask(cloud);
                    //OpenImage(shard_mask, 0, 1, 0);
                    if (flag_debug) cv::imshow("shard_mask", shard_mask);

                    /* EdgeClusterDivider */
                    EdgeClusterDivider cluster_divider_0(cloud, shard_mask, no_plane_mask);
                    cluster_divider_0.GetDividedCluster(divided_objects);
                        

                    /* TODO �ж��Ƿ���е�2�� ClusterDivider2D  */
                    if (divided_objects.size() <= 5)
                    {
                        /* ����shard_mask, ��ֹ������ظ��ж� */
                        shard_mask = cluster_divider_0.GetShardMask();
                        /* ��ʼ��������ClusterDivider2D */
                        OpenImage(no_plane_mask, 0, 2, 0);
                        ClusterDivider2D cluster_divider_1(cloud, shard_mask & no_plane_mask);
                        cluster_divider_1.GetDividedCluster(divided_objects);
                        /* �ۺ�������ȡ��ʽ�Ľ�� */
                        mask = cluster_divider_0.GetMask() | cluster_divider_1.GetMask();
                    }
                    else 
                    {
                        /* �Ե�1�����Ϊ׼ */
                        mask = cluster_divider_0.GetMask();
                    }

                    if (flag_integral)
                    {
                        /* �ۻ� */
                        for (int i=0; i<integral_mask.rows; ++i)
                            for (int j=0; j<integral_mask.cols; ++j)
                            {
                                int t = integral_mask.at<uchar>(i,j);
                                t = t * count;
                                if (mask.at<uchar>(i,j)) t+=255;
                                t /= (count+1);
                                if (t>255) t=255;
                                integral_mask.at<uchar>(i,j) = t;

                                mask.at<uchar>(i,j) = (t>128)? 255:0;
                            }
                        ++count;
                        std::cout<<count<<"\t";
                        if (count == MAX_COUNT)
                        {
                            //lock_mask = mask.clone();
                            flag_lock = true;
                            flag_integral = false;
                            count = 0;
                            std::cout<<std::endl<<"����ģʽ�ѽ���. �����ɰ��ѳ�ʼ��. ����ɰ�������(�ѽ���l״̬). "<<std::endl<<std::endl;
                        }
                        cv::imshow("integral",integral_mask);
                    }
                }

                /* ��ʾ���� */
                PointCloudPtr display_cloud;
                if (flag_all) display_cloud = GetCloudFromMask(mask_all, cloud);
                         else display_cloud = GetCloudFromMask(mask, cloud);
                if( !viewer->updatePointCloud( display_cloud, "cloud" ) )
                    viewer->addPointCloud(display_cloud, "cloud");

                if (flag_color)
                {
                    /* ��ȡ��άBGRͼƬ img */
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

                    /* �����ά���˽�� img */
                    img_result = img.clone();
                    for (int i=0; i<mask.rows; ++i)
                        for (int j=0; j<mask.cols; ++j)
                            if (! mask.at<uchar>(i,j))
                            {
                                cv::Vec3b & point = img_result.at<cv::Vec3b>(i,j);
                                /* �������� �Ҷ�, �䰵 */
                                point[0] = point[1] = point[2] = 
                                    (point[0]+point[1]+point[2])/6;
                            }
                    cv::imshow("2D Result", img_result);
                }

                if (flag_draw_box && !flag_lock)
                {
                    viewer->removeAllShapes();
                    if (flag_integral)
                    {
                        ClusterDivider2D cd(cloud, mask);
                        divided_objects.clear();
                        cd.GetDividedCluster(divided_objects);
                        for (int i=0; i<divided_objects.size(); ++i)
                            divided_objects[i].DrawBoundingBox(*viewer, i);
                        std::cout<<"�ҵ�"<<divided_objects.size()<<"��Ŀ��."<<std::endl;
                    }
                    else
                    {
                        for (int i=0; i<divided_objects.size(); ++i)
                            divided_objects[i].DrawBoundingBox(*viewer, i);
                        std::cout<<"�ҵ�"<<divided_objects.size()<<"��Ŀ��."<<std::endl;
                    }
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