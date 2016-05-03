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
#include "windows.h"

#include <boost/make_shared.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>

#include "pointcloud_filter/2dmask_filter.h"
#include "pointcloud_filter/rgbd_plane_detector.h"
#include "pointcloud_recognition/2d_cluster_divider.h"
#include "pointcloud_recognition/edge_cluster_divider.h"
#include "common.h"

using namespace tinker::vision;
using namespace std;

/* 显示点云, 以及键盘控制的各个开关 */
pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer( "3D Object Detector" ) );
bool flag_all = false;      //a
bool flag_paused = false;   //p
bool flag_draw_box = false; //d
bool flag_debug = false;    //b
bool flag_color = false;    //v
bool flag_integral = false; //i
bool flag_lock = false;     //l
bool flag_viewpoint = false;//z
bool flag_sigma = false;    //x


/* 特别的, 关于积分mask的设定 */
cv::Mat integral_mask(424, 512, CV_8UC1, cv::Scalar(0));
int count_frames = 0;
const int MAX_COUNT = 200;

/* 处理过程中的各个关键蒙版 */
//cv::Mat lock_mask(424, 512, CV_8UC1, cv::Scalar(0));
cv::Mat mask(424, 512, CV_8UC1, cv::Scalar(0));
cv::Mat shard_mask(424, 512, CV_8UC1, cv::Scalar(0));
cv::Mat no_plane_mask(424, 512, CV_8UC1, cv::Scalar(0));
const cv::Mat mask_all(424, 512, CV_8UC1, cv::Scalar(255));
cv::Mat mask_downsample(424, 512, CV_8UC1, cv::Scalar(0));
/* BGR图像 */
cv::Mat img(424, 512, CV_8UC3, cv::Scalar(0,0,0));
cv::Mat img_result(424, 512, CV_8UC3, cv::Scalar(0,0,0));
/* 目标vector */
vector<ObjectCluster> divided_objects;

/* 用于累计的点云 */
PointCloudPtr cloud_sigma(new PointCloud);
Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();

/* 从蓝牙获取的信息通过内存共享获取 */
LPCWSTR strMapName = L"BluetoothAngleSharedMemory";
HANDLE hMap = NULL;
LPVOID pBuffer = NULL;
bool MemShareIntialize()
{
    hMap = OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, strMapName);
    if (NULL == hMap)
    {
        cout<< "警告, 没有接收到姿态信号!" <<endl;
        return false;
    }
    cout << "姿态信号已连接. " <<endl;
    return true;
}
void MemShareGet()
{
    if (NULL == hMap)
    {
        flag_viewpoint = false;
        cout << "姿态信息意外掉线!(hMap) 请检查蓝牙与串口模块. " << endl;
        return;
    }
    pBuffer = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (NULL == pBuffer)
    {
        hMap = NULL;
        flag_viewpoint = false;
        cout << "姿态信息意外掉线!(pBuffer) 请检查蓝牙与串口模块. " << endl;
        return;
    }
    
    double* d = (double*)pBuffer;
    double alpha, beta, gamma;
    /*PointCloudPtr new_cloud(new PointCloud); */
    alpha = d[0]*3.1415926/180;
    beta  = d[1]*3.1415926/180;
    gamma = d[2]*3.1415926/180;
    
    transform_matrix << 
        cos(beta)*cos(gamma)  , -cos(alpha)*sin(beta)*cos(gamma)-sin(alpha)*sin(gamma) , -sin(alpha)*sin(beta)*cos(gamma)+cos(alpha)*sin(gamma)  , 0 ,
        sin(beta)             , cos(beta)*cos(beta)                                    , sin(alpha)*cos(beta)                                  , 0 ,
        -cos(beta)*sin(gamma) , cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma)  , sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma) , 0 ,
        0                     , 0                                                      , 0                                                      , 1 ;

    /*pcl::transformPointCloud(*cloud, *new_cloud, transform_matrix);
    cloud = new_cloud;*/

    //cout << "读取共享内存数据：" << d[0]<<"\t"<<d[1]<<"\t"<<d[2] << endl;

}
void RotateCloud(PointCloudPtr& cloud)
{
    PointCloudPtr new_cloud(new PointCloud);
    pcl::transformPointCloud(*cloud, *new_cloud, transform_matrix);
    cloud = new_cloud;
}


/* 键盘响应 */
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
            cout<<"已停止接收点云信息. 已锁定结果蒙版. 已暂停对蒙版的计算."<<endl;
        }
        else
        {
            flag_lock = false;
            cout<<"开始接收点云信息. 已取消对结果蒙版的锁定."<<endl;
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
            cout<<"找到"<<divided_objects.size()<<"个目标."<<endl;
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
            cout<<"已进入积分模式, 开始采集"<<MAX_COUNT<<"帧图像."<<endl;
        }
        else
        {
            count_frames = 0;
            cout<<"已退出积分模式. "<<endl;
        }
    }
    if (event.getKeySym() == "l" && event.keyDown())
    {
        if (flag_paused)
        {
            cout<<"请先退出p状态(暂停接收点云), 再尝试锁定结果蒙版! "<<endl;
        }
        else
        {
            flag_lock = !flag_lock;
        
            if (flag_lock)
                cout<<"已锁定结果蒙版. 已暂停对蒙版的计算."<<endl;
            else
                cout<<"已取消对结果蒙版的锁定. "<<endl;
        }
    }

    if (event.getKeySym() == "z" && event.keyDown())
    {
        if (flag_viewpoint)
        {
            flag_viewpoint = false;
            cout<< "已停止姿态追踪." <<endl;
        }
        else
        {
            if (MemShareIntialize())
            {
                flag_viewpoint = true;
                cout << "已开始姿态追踪." <<endl;
            }
            else
            {
                flag_viewpoint = false;
                cout << "q操作失败, 请检查蓝牙与串口模块. " <<endl;
            }
        }
    }
    if (event.getKeySym() == "x" && event.keyDown())
    {
        if (flag_viewpoint)
        {
            if (flag_sigma)
            {
                flag_sigma = false;
                pcl::io::savePCDFileBinary("sigma.pcd", *cloud_sigma);
                
                PointCloudPtr null_cloud(new PointCloud);
                cloud_sigma = null_cloud;

                cout<< "场景采集结束, 已保存到\"sigma.pcd\"." <<endl;
            }
            else
            {   
                flag_sigma = true;
                cout<< "开始抽样全方位场景." <<endl;
            }
        }
        else
        {
            flag_sigma = false;
            cout<< "请先开启姿态跟踪(z), 再尝试全方向(x)." << endl;
        }
    }

    if (event.getKeySym() == "r" && event.keyDown())
    {
        viewer->setCameraPosition( 0.0, 0.0, -2.5, 0.0, 0.0, 0.0 );
        cout<<"视角已重置."<<endl;
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

    /* flag_sigma配置 */
    for (int i=5; i<424; i+=5)
        for (int j=5; j<512; j+=5)
            mask_downsample.at<uchar>(i,j)=255;

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
                    viewer->spinOnce(200);
                    continue;
                }

                if (flag_viewpoint)
                {
                    MemShareGet();
                }

                if (!flag_lock)
                {
                    /* 处理点云 */
                    divided_objects.clear();

                    /* 去平面 */
                    no_plane_mask = GetNoPlaneMask(cloud);
                    OpenImage(no_plane_mask, 0, 2, 0);
                    if (flag_debug) cv::imshow("no_plane_mask", no_plane_mask);

                    /* 边缘拉伸点 */
                    shard_mask = ~GetShardMask(cloud);
                    //OpenImage(shard_mask, 0, 1, 0);
                    if (flag_debug) cv::imshow("shard_mask", shard_mask);

                    /* EdgeClusterDivider */
                    EdgeClusterDivider cluster_divider_0(cloud, shard_mask, no_plane_mask);
                    cluster_divider_0.GetDividedCluster(divided_objects);
                        

                    /* TODO 判定是否进行第2步 ClusterDivider2D  */
                    if (divided_objects.size() <= 5)
                    {
                        /* 更新shard_mask, 防止区域的重复判定 */
                        shard_mask = cluster_divider_0.GetShardMask();
                        /* 初始化并进行ClusterDivider2D */
                        OpenImage(no_plane_mask, 0, 2, 0);
                        ClusterDivider2D cluster_divider_1(cloud, shard_mask & no_plane_mask);
                        cluster_divider_1.GetDividedCluster(divided_objects);
                        /* 综合两种提取方式的结果 */
                        mask = cluster_divider_0.GetMask() | cluster_divider_1.GetMask();
                    }
                    else 
                    {
                        /* 以第1步结果为准 */
                        mask = cluster_divider_0.GetMask();
                    }

                    if (flag_integral)
                    {
                        /* 累积 */
                        for (int i=0; i<integral_mask.rows; ++i)
                            for (int j=0; j<integral_mask.cols; ++j)
                            {
                                int t = integral_mask.at<uchar>(i,j);
                                t = t * count_frames;
                                if (mask.at<uchar>(i,j)) t+=255;
                                t /= (count_frames+1);
                                if (t>255) t=255;
                                integral_mask.at<uchar>(i,j) = t;

                                mask.at<uchar>(i,j) = (t>128)? 255:0;
                            }
                        ++count_frames;
                        cout<<count_frames<<"\t";
                        if (count_frames == MAX_COUNT)
                        {
                            //lock_mask = mask.clone();
                            flag_lock = true;
                            flag_integral = false;
                            count_frames = 0;
                            cout<<endl<<"积分模式已结束. 积分蒙版已初始化. 结果蒙版已锁定(已进入l状态). "<<endl<<endl;
                        }
                        cv::imshow("integral",integral_mask);
                    }
                }

                /* 显示点云 */
                PointCloudPtr display_cloud;
                if (flag_all) display_cloud = GetCloudFromMask(mask_all, cloud);
                         else display_cloud = GetCloudFromMask(mask, cloud);
                if (flag_viewpoint)
                {
                    RotateCloud(display_cloud);

                    if (flag_sigma)
                    {
                        PointCloudPtr cloud_downsample = GetCloudFromMask(mask_downsample, cloud);
                        
                        RotateCloud(cloud_downsample);

                        *cloud_sigma += *cloud_downsample;
                        *display_cloud += *cloud_sigma;
                    }
                }
                if( !viewer->updatePointCloud( display_cloud, "cloud" ) )
                    viewer->addPointCloud(display_cloud, "cloud");

                if (flag_color)
                {
                    /* 获取二维BGR图片 img */
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

                    /* 处理二维过滤结果 img */
                    img_result = img.clone();
                    for (int i=0; i<mask.rows; ++i)
                        for (int j=0; j<mask.cols; ++j)
                            if (! mask.at<uchar>(i,j))
                            {
                                cv::Vec3b & point = img_result.at<cv::Vec3b>(i,j);
                                /* 背景部分 灰度, 变暗 */
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
                        cout<<"找到"<<divided_objects.size()<<"个目标."<<endl;
                    }
                    else
                    {
                        for (int i=0; i<divided_objects.size(); ++i)
                            divided_objects[i].DrawBoundingBox(*viewer, i);
                        cout<<"找到"<<divided_objects.size()<<"个目标."<<endl;
                    }
                }       
            }
            else
            {
                cout<<" Warning: No cloud received!"<<endl;
            }
        }
    }

    // Stop Grabber
    grabber->stop();

    return 0;
}