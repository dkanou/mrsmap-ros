/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 24.09.2012
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

#ifndef VISUALIZATION_MAP_H_
#define VISUALIZATION_MAP_H_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/ros/conversions.h>
#include "pcl_ros/point_cloud.h" //Important
#include "pcl/visualization/pcl_visualizer.h"
#include <sensor_msgs/PointCloud2.h>
#include <ros/ros.h>

#include <string>

namespace mrsmap {

class Viewer {
public:
	Viewer(ros::NodeHandle* n = NULL);
	virtual ~Viewer();

	void spinOnce();

	virtual void keyboardEventOccurred( const pcl::visualization::KeyboardEvent &event, void* data );

	void displayPointCloud( const std::string& name, const pcl::PointCloud< pcl::PointXYZRGBA >::Ptr& cloud, int pointSize = 5 );
	void advertisePointCloud2( const pcl::PointCloud< pcl::PointXYZRGBA >::Ptr& cloud );
	void displayPose( const Eigen::Matrix4d& pose );
	void displayCorrespondences( const std::string& name, const pcl::PointCloud< pcl::PointXYZ >::Ptr& cloud1, const pcl::PointCloud< pcl::PointXYZ >::Ptr& cloud2 );

	int selectedDepth;
	int selectedViewDir;
	bool processFrame;
	bool displayScene;
	bool displayMap;
	bool displayCorr;
	bool displayAll;
	bool recordFrame;
	bool forceRedraw;

	bool is_running;
	bool close;

	boost::shared_ptr< pcl::visualization::PCLVisualizer > viewer;

	int shapeIdx;
	std::vector< int > currShapes;

	ros::NodeHandle* nh;
	ros::Publisher pub;

};

}


#endif /* VISUALIZATION_MAP_H_ */

