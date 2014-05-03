/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 02.01.2012
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

#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

#include <pcl/common/transforms.h>

#include <Eigen/Core>

#include <boost/algorithm/string.hpp>

#include <boost/thread/thread.hpp>
#include "pcl/common/common_headers.h"
#include "pcl/visualization/pcl_visualizer.h"

#include <mrsmap/slam/slam.h>
#include <mrsmap/visualization/visualization_slam.h>
#include <mrsmap/utilities/utilities.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace mrsmap;

//typedef MultiResolutionColorSurfelMap MultiResolutionSurfelMap;

// parses Juergen Sturm's datasets (tgz archives + timestamp associations)
// simply takes the base path of the dataset

class EvaluateSLAM {
public:

	EvaluateSLAM(  int argc, char** argv  )
			: viewer_( &slam_ ) {


		po::options_description desc("Allowed options");
		desc.add_options()
		    ("help,h", "help")
		    ("inputpath,i", po::value<std::string>(&path_)->default_value("."), "path to input data")
		    ("maxresolution,r", po::value<double>(&min_resolution_)->default_value(0.0125f), "maximum resolution")
		    ("skippastframes,k", po::value<bool>(&skip_past_frames_)->default_value(false), "skip past frames for real-time evaluation")
		;

    	po::variables_map vm;
    	po::store(po::parse_command_line(argc, argv, desc), vm);
    	po::notify(vm);

    	if( vm.count("help") || vm.count("h") ) {
    		std::cout << desc << "\n";
    		exit(0);
    	}

		max_radius_ = 30.f;

		imageAllocator_ = boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator >( new MultiResolutionSurfelMap::ImagePreAllocator() );

		for( int i = 0; i < 2; i++ ) {
			treeNodeAllocator_[ i ] = boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > >(
					new spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue >( 1000 ) );
		}

		graphChanged_ = true;

	}

	class PoseInfo {
	public:
		PoseInfo( const std::string& time, int id, const Eigen::Matrix4d tf )
				: stamp( time ), referenceID( id ), transform( tf ) {
		}
		~PoseInfo() {
		}

		std::string stamp;
		int referenceID;
		Eigen::Matrix4d transform;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		;
	};

	void evaluate() {

		float register_start_resolution = min_resolution_;
		const float register_stop_resolution = 32.f * min_resolution_;

		// parse associations.txt
		std::ifstream assocFile( ( path_ + std::string( "/associations.txt" ) ).c_str() );

		Eigen::Matrix4f totalTransform;
		totalTransform.setIdentity();

		lastTransform_.setIdentity();

		int count = -1;

		std::vector< PoseInfo, Eigen::aligned_allocator< PoseInfo > > trajectoryEstimate;

		double nextTime = 0;

		while( assocFile.good() ) {

			// read in line
			char lineCStr[ 1024 ];
			assocFile.getline( lineCStr, 1024, '\n' );

			count++;

			std::string lineStr( lineCStr );

			// split line at blanks
			std::vector< std::string > entryStrs;
			boost::split( entryStrs, lineStr, boost::is_any_of( "\t " ) );

			// parse entries, load images, generate point cloud, process images...
			if( entryStrs.size() == 4 ) {

				while( !viewer_.processFrame && viewer_.is_running ) {
					usleep( 10 );
				}


				double stamp = 0.0;
				std::stringstream sstr;
				sstr << entryStrs[0];
				sstr >> stamp;

				if( skip_past_frames_ && nextTime > stamp ) {
					std::cout << "================= SKIP =================\n";
					continue;
				}


				// load images
				cv::Mat depthImg = cv::imread( path_ + "/" + entryStrs[ 1 ], CV_LOAD_IMAGE_ANYDEPTH );
				cv::Mat rgbImg = cv::imread( path_ + "/" + entryStrs[ 3 ], CV_LOAD_IMAGE_ANYCOLOR );

				// extract point cloud from image pair
				pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud( new pcl::PointCloud< pcl::PointXYZRGBA >() );
				imagesToPointCloud( depthImg, rgbImg, entryStrs[ 0 ], cloud );

	    		// measure time to skip frames
	    		pcl::StopWatch stopwatch;
	    		stopwatch.reset();

	    		unsigned int numEdges = slam_.optimizer_->edges().size();
				unsigned int numVertices = slam_.optimizer_->vertices().size();
				unsigned int referenceID = slam_.referenceKeyFrameId_;

				bool retVal = slam_.addImage( rgbImg, cloud, register_start_resolution, register_stop_resolution, min_resolution_, false );

				double deltat = stopwatch.getTimeSeconds() * 1000.0;
				std::cout << "slam iteration took: " << deltat << "\n";

				nextTime = stamp + 0.001 * deltat;

				if( retVal ) {
					// store relative translation to reference keyframe
					trajectoryEstimate.push_back( PoseInfo( entryStrs[ 0 ], slam_.referenceKeyFrameId_, slam_.lastTransform_ ) );
				}

				if( slam_.optimizer_->vertices().size() != numVertices || slam_.optimizer_->edges().size() != numEdges || slam_.referenceKeyFrameId_ != referenceID )
					graphChanged_ = true;

				if( slam_.optimizer_->vertices().size() > 0 ) {
					g2o::VertexSE3* v_ref = dynamic_cast< g2o::VertexSE3* >( slam_.optimizer_->vertex( slam_.keyFrames_[ slam_.referenceKeyFrameId_ ]->nodeId_ ) );
					Eigen::Matrix4d pose_ref = v_ref->estimate().matrix();

					viewer_.displayPose( pose_ref * slam_.lastTransform_ );

				}

				if( !viewer_.is_running )
					exit( -1 );

				if( graphChanged_ || viewer_.forceRedraw ) {
					viewer_.visualizeSLAMGraph();
					viewer_.forceRedraw = false;
				}

//				if( recordFrame ) {
//					static unsigned int frameId = 0;
//					char frameStr[255];
//					sprintf( frameStr, "/home/stueckler/.ros/slam%05d.png", frameId++ );
//					viewer->saveScreenshot( frameStr );
//				}

				viewer_.spinOnce();
				usleep( 1000 );

			}

		}

		if( graphChanged_ || viewer_.forceRedraw ) {
			viewer_.visualizeSLAMGraph();
			viewer_.forceRedraw = false;
			viewer_.spinOnce();
		}

		// dump pose estimates to file
		std::ofstream outFile1( ( path_ + "/" + std::string( "slam_result.txt" ) ).c_str() );
		outFile1 << "# minres: " << min_resolution_ << ", max depth: " << max_radius_ << "\n";
		for( unsigned int i = 0; i < trajectoryEstimate.size(); i++ ) {

			g2o::VertexSE3* v_curr = dynamic_cast< g2o::VertexSE3* >( slam_.optimizer_->vertex( slam_.keyFrames_[ trajectoryEstimate[ i ].referenceID ]->nodeId_ ) );
			Eigen::Matrix4d vtransform = v_curr->estimate().matrix();

			Eigen::Matrix4d transform = vtransform * trajectoryEstimate[ i ].transform;

			Eigen::Quaterniond q( Eigen::Matrix3d( transform.block< 3, 3 >( 0, 0 ) ) );
			outFile1 << trajectoryEstimate[ i ].stamp << " " << transform( 0, 3 ) << " " << transform( 1, 3 ) << " " << transform( 2, 3 ) << " " << q.x() << " " << q.y() << " " << q.z() << " "
					<< q.w() << "\n";

		}

	}

public:

	std::string path_;

	SLAM slam_;

	Eigen::Matrix4d lastTransform_;

	double min_resolution_, max_radius_;

	bool skip_past_frames_;

	boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator > imageAllocator_;
	boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > > treeNodeAllocator_[ 2 ];

	ViewerSLAM viewer_;

	bool graphChanged_;

};

int main( int argc, char** argv ) {

	EvaluateSLAM ev( argc, argv );
	ev.evaluate();

	while( ev.viewer_.is_running ) {

		if( ev.viewer_.forceRedraw ) {
			ev.viewer_.visualizeSLAMGraph();
			ev.viewer_.forceRedraw = false;
		}

		ev.viewer_.spinOnce();
		usleep( 1000 );
	}

	return 0;
}

