/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 12.12.2011
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of University of Bonn, Computer Science Institute 
 *     VI nor the names of its contributors may be used to endorse or 
 *     promote products derived from this software without specific 
 *     prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/registration/transforms.h>

#include <mrsmap/map/multiresolution_csurfel_map.h>
#include <mrsmap/slam/slam.h>

#include <pcl/io/openni_grabber.h>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include "pcl/common/common_headers.h"
#include "pcl/visualization/pcl_visualizer.h"

#include <mrsmap/visualization/visualization_slam.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>

using namespace mrsmap;

boost::mutex mutex;

int selectedDepth = 10;
int selectedViewDir = -1;
bool displayRandom = false;
bool displayAll = true;
bool displayMap = true;
bool integrateMeasurements = true;
bool graphChanged = true;
bool forceRedraw = false;

class SLAMNode {
public:
	SLAMNode(ros::NodeHandle* nh)
			: viewer_( &slam_, nh) {
	}

	~SLAMNode() {
	}

	void dataCallback( const pcl::PointCloud< pcl::PointXYZRGBA >::ConstPtr& pointCloudIn ) {

		if( !integrateMeasurements )
			return;

		static int counter = 0;
		if( counter++ < 100 )
			return;

		if( pointCloudIn->points.size() == 0 )
			return;

		const float register_start_resolution = 0.8f;
		const float register_stop_resolution = 0.0125f;

		const float min_resolution = 0.0125f;

		boost::lock_guard< boost::mutex > lock( mutex );

		int numEdges = slam_.optimizer_->edges().size();
		int numVertices = slam_.optimizer_->vertices().size();
		int referenceID = slam_.referenceKeyFrameId_;

		cv::Mat img_rgb;
		slam_.addImage( img_rgb, pointCloudIn, register_start_resolution, register_stop_resolution, min_resolution );

		if( slam_.optimizer_->vertices().size() != numVertices || slam_.optimizer_->edges().size() != numEdges || slam_.referenceKeyFrameId_ != referenceID )
			graphChanged = true;

	}

	SLAM slam_;
	ViewerSLAM viewer_;

};

int main( int argc, char** argv ) {
	ros::init(argc, argv, "mrsslam");
	ros::NodeHandle nh;

	//create the root node in tf tree
	tf::TransformBroadcaster cam_base_br;
	tf::Transform cam_base_tf;
	cam_base_tf.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
	cam_base_tf.setRotation(tf::Quaternion(0.0, 0.0, 0.0, 1.0));

	// create a new grabber for OpenNI devices
	SLAMNode slamNode(&nh);
	pcl::OpenNIGrabber* interface = new pcl::OpenNIGrabber( "", pcl::OpenNIGrabber::OpenNI_QVGA_30Hz, pcl::OpenNIGrabber::OpenNI_QVGA_30Hz );
	boost::function< void( const pcl::PointCloud< pcl::PointXYZRGBA >::ConstPtr& ) > f = boost::bind( &SLAMNode::dataCallback, &slamNode, _1 );
	boost::signals2::connection c = interface->registerCallback( f );
	// start receiving point clouds
	interface->start();

	while( !slamNode.viewer_.viewer->wasStopped() || ros::ok()) {
		if( graphChanged || forceRedraw ) {
			boost::lock_guard< boost::mutex > lock( mutex );
			slamNode.viewer_.visualizeSLAMGraph();
			slamNode.viewer_.forceRedraw = false;
			slamNode.viewer_.spinOnce();
		}
		cam_base_br.sendTransform(tf::StampedTransform(cam_base_tf, ros::Time::now(), "camera_base", "kinect_base"));
		ros::spinOnce();
		usleep( 1000 );
	}

	// stop the grabber
	interface->stop();

	return 0;

}

