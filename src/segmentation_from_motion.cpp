#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
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
#include <omp.h>
ros::Publisher pub_result_cloud_fast_, pub_result_cloud_sac_, pub_result_cloud_sac2_, pub_result_cloud_sac3_, pub_result_cloud_sac4_, pub_result_cloud_seed_, pub_result_pose_seed_;

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{
  // Create a container for the data.
  pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
  pcl::PointCloud<pcl::PointXYZRGBNormal> cloud_fast;
  pcl::fromROSMsg(*input, cloud);
  for (size_t i=0; i < cloud.points.size(); i++) {
    pcl::PointXYZRGBNormal point_temp = cloud.points[i];
    if (sqrt(point_temp.normal_x*point_temp.normal_x+point_temp.normal_y*point_temp.normal_y+point_temp.normal_z*point_temp.normal_z) > 0.02) {
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
  int mirror_num = 0;
  if (cloud_fast.points.size() > 50) {
    omp_lock_t writelock;
    omp_init_lock(&writelock);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i=0; i < 2000; i++) {
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
      if (R.determinant() < 0.0) {mirror_num++; continue;}
      Eigen::Vector3f t = v_after_ave - R * v_before_ave;
      unsigned int inliers = 0;
      for (size_t j=0; j < cloud_fast.points.size(); j++) {
        pcl::PointXYZRGBNormal point_temp = cloud_fast.points[j];
        Eigen::Vector3f v_before_temp = Eigen::Vector3f(point_temp.x, point_temp.y, point_temp.z);
        Eigen::Vector3f v_after_temp = v_before_temp + Eigen::Vector3f(point_temp.normal_x, point_temp.normal_y, point_temp.normal_z);
        Eigen::Vector3f v_after_temp_estimated = R * v_before_temp + t;
        float error = (v_after_temp_estimated - v_after_temp).norm();
        if (error < 0.03){
          inliers ++;
        }
      }
      omp_set_lock(&writelock);
      if (max_inliers < inliers) {
        max_inliers = inliers;
        R_best = R;
        t_best = t;
        for (size_t j=0; j < 3; j++) {
          point_best.points[j]=cloud_fast.points[rand_nums[j]];
        }
      }
      omp_unset_lock(&writelock);
    }
    ROS_INFO("Done sac, total moving: %d, max_i: %d", cloud_fast.points.size(), max_inliers);
    apply_sac = true;
    ROS_INFO("determinant %f", R_best.determinant());
    //ROS_INFO("mirror num %d", mirror_num);
  } else{
    ROS_INFO("No moving points");
    apply_sac = false;
  }
  sensor_msgs::PointCloud2 ros_out;
  pcl::toROSMsg(cloud_fast, ros_out);
  ros_out.header = input->header;
  pub_result_cloud_fast_.publish(ros_out); 
  if (apply_sac){
    pcl::PointCloud<pcl::PointXYZRGBNormal> cloud_sac, cloud_sac2, cloud_sac3, cloud_sac4;
    for (size_t j=0; j < cloud.points.size(); j++) {
      pcl::PointXYZRGBNormal point_temp = cloud.points[j];
      Eigen::Vector3f v_before_temp = Eigen::Vector3f(point_temp.x, point_temp.y, point_temp.z);
      Eigen::Vector3f v_after_temp = v_before_temp + Eigen::Vector3f(point_temp.normal_x, point_temp.normal_y, point_temp.normal_z);
      Eigen::Vector3f v_after_temp_estimated = R_best * v_before_temp + t_best;
      float error = (v_after_temp_estimated - v_after_temp).norm();
      if (error < 0.03){
        cloud_sac.points.push_back(cloud.points[j]);

        pcl::PointXYZRGBNormal point_temp2;
        point_temp2.x = v_after_temp_estimated[0];
        point_temp2.y = v_after_temp_estimated[1];
        point_temp2.z = v_after_temp_estimated[2];
        cloud_sac2.points.push_back(point_temp2);
        
        v_after_temp_estimated = R_best * v_after_temp_estimated + t_best;
        pcl::PointXYZRGBNormal point_temp3;
        point_temp3.x = v_after_temp_estimated[0];
        point_temp3.y = v_after_temp_estimated[1];
        point_temp3.z = v_after_temp_estimated[2];
        cloud_sac3.points.push_back(point_temp3);

        v_after_temp_estimated = R_best * v_after_temp_estimated + t_best;
        pcl::PointXYZRGBNormal point_temp4;
        point_temp4.x = v_after_temp_estimated[0];
        point_temp4.y = v_after_temp_estimated[1];
        point_temp4.z = v_after_temp_estimated[2];
        cloud_sac4.points.push_back(point_temp4);
        // if (j == 20) {
        //   Eigen::Vector3f v_a_b = v_after_temp_estimated4;
        //   for (int k=0; k<10; k++){
        //     v_a_b = R_best * v_a_b + t_best;
        //     ROS_INFO("v_a_b, x:%f, y:%f, z:%f", v_a_b[0], v_a_b[1], v_a_b[2]);
        //   }
        // }
      }
    }
    cloud_sac.width=cloud_sac2.width=cloud_sac3.width=cloud_sac4.width=cloud_sac.points.size();
    cloud_sac.height=cloud_sac2.height=cloud_sac3.height=cloud_sac4.height=1;

    pcl::toROSMsg(cloud_sac, ros_out);
    ros_out.header = input->header;
    pub_result_cloud_sac_.publish(ros_out);

    pcl::toROSMsg(cloud_sac2, ros_out);
    ros_out.header = input->header;
    pub_result_cloud_sac2_.publish(ros_out);

    pcl::toROSMsg(cloud_sac3, ros_out);
    ros_out.header = input->header;
    pub_result_cloud_sac3_.publish(ros_out);

    pcl::toROSMsg(cloud_sac4, ros_out);
    ros_out.header = input->header;
    pub_result_cloud_sac4_.publish(ros_out);

    pcl::toROSMsg(point_best, ros_out);
    ros_out.header = input->header;
    pub_result_cloud_seed_.publish(ros_out);
    
    visualization_msgs::MarkerArray marker_array_best;
    for (int i=0; i<3; i++) {
      visualization_msgs::Marker marker_best;
      geometry_msgs::Point point;

      marker_best.header = input->header;
      marker_best.type = visualization_msgs::Marker::ARROW;
      point.x = point_best.points[i].x;
      point.y = point_best.points[i].y;
      point.z = point_best.points[i].z;
      marker_best.points.push_back(point);
      point.x = point_best.points[i].x + point_best.points[i].normal_x;
      point.y = point_best.points[i].y + point_best.points[i].normal_y;
      point.z = point_best.points[i].z + point_best.points[i].normal_z;
      marker_best.points.push_back(point);
      marker_best.id = i;
      marker_best.ns = std::string("debug_point_array: ");
      marker_best.scale.x = 0.01;
      marker_best.scale.y = 0.02;
      marker_best.scale.z = 0.01;
      marker_best.color.r = 1.0;
      marker_best.color.a = 1.0;
      marker_array_best.markers.push_back(marker_best);
    }
    pub_result_pose_seed_.publish(marker_array_best);
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
  pub_result_cloud_sac2_ = nh.advertise<sensor_msgs::PointCloud2> ("output_sac2", 1);
  pub_result_cloud_sac3_ = nh.advertise<sensor_msgs::PointCloud2> ("output_sac3", 1);
  pub_result_cloud_sac4_ = nh.advertise<sensor_msgs::PointCloud2> ("output_sac4", 1);
  pub_result_cloud_seed_ = nh.advertise<sensor_msgs::PointCloud2> ("output_seed", 1);
  pub_result_cloud_fast_ = nh.advertise<sensor_msgs::PointCloud2> ("output_fast", 1);
  pub_result_pose_seed_ = nh.advertise<visualization_msgs::MarkerArray> ("output_marker_seed_", 1);

  // Spin
  ros::spin ();
}
