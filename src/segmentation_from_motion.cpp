#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <kdl/utilities/svd_eigen_HH.hpp>
#include <math.h>
#include <algorithm>
#include "jsk_recognition_utils/pcl_conversion_util.h"
#include <eigen_conversions/eigen_msg.h>
#include <pcl/common/transforms.h>
#include <eigen_conversions/eigen_msg.h>

ros::Publisher pub_result_cloud_fast_, pub_result_cloud_sac_, pub_result_cloud_seed_;

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  // Create a container for the data.
  pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
  pcl::PointCloud<pcl::PointXYZRGBNormal> cloud_fast;
  pcl::fromROSMsg(*input, cloud);
  for (size_t i=0; i < cloud.points.size(); i++) {
    pcl::PointXYZRGBNormal point_temp = cloud.points[i];
    if (sqrt(point_temp.normal_x*point_temp.normal_x+point_temp.normal_y*point_temp.normal_y+point_temp.normal_z*point_temp.normal_z) > 0.015) {
      cloud_fast.points.push_back(cloud.points[i]);
    }
  }

  cloud_fast.width=cloud_fast.points.size();
  cloud_fast.height=1;
  ROS_INFO("done extract fast size %d", cloud_fast.points.size());
  unsigned int max_inliers = 0;
  Eigen::Matrix3f R_best;
  Eigen::Vector3f t_best;
  pcl::PointCloud<pcl::PointXYZRGBNormal> point_best;
  point_best.points.resize(3); point_best.width = 3; point_best.height = 1;
  bool apply_sac;
  if (cloud_fast.points.size() > 50) {
    for (size_t i=0; i < 300; i++) {
      unsigned int rand_nums[3];
      while (true) {
        rand_nums[0] = rand() % cloud_fast.width;
        rand_nums[1] = rand() % cloud_fast.width;
        rand_nums[2] = rand() % cloud_fast.width;
        if (rand_nums[0] != rand_nums[1] && rand_nums[1] != rand_nums[2] && rand_nums[2] != rand_nums[0])
          break;
      }
      Eigen::Vector3f v_before[3];
      Eigen::Vector3f v_after[3];
      for (size_t j=0; j < 3; j++) {
        pcl::PointXYZRGBNormal point_temp = cloud_fast.points[rand_nums[j]];
        v_before[j] = Eigen::Vector3f(point_temp.x, point_temp.y, point_temp.z);
        v_after[j] = v_before[j] + Eigen::Vector3f(point_temp.normal_x, point_temp.normal_y, point_temp.normal_z);
      }
      Eigen::Vector3f v_before_ave = (v_before[0] + v_before[1] + v_before[2]) / 3.0;
      Eigen::Vector3f v_after_ave = (v_after[0] + v_after[1] + v_after[2]) / 3.0;
      Eigen::Vector3f v_before_relative[3];
      Eigen::Vector3f v_after_relative[3];
      Eigen::MatrixXf m_before_ave = Eigen::MatrixXf::Zero(3, 3);
      Eigen::MatrixXf m_after_ave = Eigen::MatrixXf::Zero(3, 3);
      for (size_t j=0; j < 3; j++) {
        v_before_relative[j] = v_before[j] - v_before_ave;
        v_after_relative[j] = v_after[j] - v_after_ave;
        m_before_ave.col(j) = v_before_relative[j];
        m_after_ave.col(j) = v_after_relative[j];
      }

      Eigen::Matrix3f Q = m_after_ave * m_before_ave.transpose();
      JacobiSVD<Eigen::Matrix3f> svd(Q, ComputeFullU | ComputeFullV);
      Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();
      Eigen::Vector3f t = v_after_ave - R * v_before_ave;
      unsigned int inliers = 0;
      for (size_t j=0; j < cloud_fast.points.size(); j++) {
        pcl::PointXYZRGBNormal point_temp = cloud_fast.points[j];
        Eigen::Vector3f v_before_temp = Eigen::Vector3f(point_temp.x, point_temp.y, point_temp.z);
        Eigen::Vector3f v_after_temp = v_before_temp + Eigen::Vector3f(point_temp.normal_x, point_temp.normal_y, point_temp.normal_z);
        Eigen::Vector3f v_after_temp_estimated = R * v_before_temp + t;
        float error = (v_after_temp_estimated - v_after_temp).norm();
        if (error < 0.02){
          inliers ++;
        }
      }
      if (max_inliers < inliers) {
        max_inliers = inliers;
        R_best = R;
        t_best = t;
        for (size_t j=0; j < 3; j++) {
          point_best.points[j]=cloud_fast.points[rand_nums[j]];
        }
      }
    }
    ROS_INFO("Done sac, total moving: %d, max_i: %d", cloud_fast.points.size(), max_inliers);
    apply_sac = true;
  } else{
    ROS_INFO("No moving points");
    apply_sac = false;
  }
  pcl::PointCloud<pcl::PointXYZRGBNormal> cloud_sac;
  for (size_t j=0; j < cloud.points.size(); j++) {
    pcl::PointXYZRGBNormal point_temp = cloud.points[j];
    Eigen::Vector3f v_before_temp = Eigen::Vector3f(point_temp.x, point_temp.y, point_temp.z);
    Eigen::Vector3f v_after_temp = v_before_temp + Eigen::Vector3f(point_temp.normal_x, point_temp.normal_y, point_temp.normal_z);
    Eigen::Vector3f v_after_temp_estimated = R_best * v_before_temp + t_best;
    float error = (v_after_temp_estimated - v_after_temp).norm();
    if (error < 0.02){
      cloud_sac.points.push_back(cloud.points[j]);
    }
  }
  cloud_sac.width=cloud_sac.points.size();
  cloud_sac.height=1;
  sensor_msgs::PointCloud2 ros_out;
  pcl::toROSMsg(cloud_fast, ros_out);
  ros_out.header = input->header;
  pub_result_cloud_fast_.publish(ros_out); 
  if (apply_sac){
    pcl::toROSMsg(cloud_sac, ros_out);
    ros_out.header = input->header;
    pub_result_cloud_sac_.publish(ros_out);
    pcl::toROSMsg(point_best, ros_out);
    ros_out.header = input->header;
    pub_result_cloud_seed_.publish(ros_out);
  }
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "segmentation_from_motion");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("input", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub_result_cloud_sac_ = nh.advertise<sensor_msgs::PointCloud2> ("output_sac", 1);
  pub_result_cloud_seed_ = nh.advertise<sensor_msgs::PointCloud2> ("output_seed", 1);
  pub_result_cloud_fast_ = nh.advertise<sensor_msgs::PointCloud2> ("output_fast", 1);

  // Spin
  ros::spin ();
}
