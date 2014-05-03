/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 20.12.2011
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
#include <pcl/common/transforms.h>

#include <opencv2/opencv.hpp>

#include <Eigen/Core>

#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/surface/convex_hull.h>

#include <mrsmap/map/multiresolution_csurfel_map.h>
#include <mrsmap/registration/multiresolution_csurfel_registration.h>

#include <boost/thread/thread.hpp>
#include "pcl/common/common_headers.h"
#include "pcl/visualization/pcl_visualizer.h"

#include "pcl/common/centroid.h"
#include "pcl/common/eigen.h"

#include <mrsmap/slam/slam.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <mrsmap/visualization/visualization_slam.h>
#include <mrsmap/utilities/utilities.h>

using namespace mrsmap;

class ChessboardInfo {
public:

	ChessboardInfo() {
		initPose();
	}

	~ChessboardInfo() {
	}

	void initPose() {
		cv::Mat R = cv::Mat::eye( 3, 3, CV_64FC1 );
		cv::Rodrigues( R, rotation );
		translation = cv::Mat::zeros( 3, 1, CV_64FC1 );
		trackInitialized = false;
	}

	void initCorners() {

		corners.clear();
		for( int i = 0; i < size.height; i++ )
			for( int j = 0; j < size.width; j++ )
				corners.push_back( cv::Point3f( float( j * squareSize ), float( i * squareSize ), 0.f ) );

	}

	cv::Size size;
	double squareSize;
	std::vector< cv::Point3f > corners;
	cv::Mat rotation, translation;
	bool trackInitialized;

};

struct MouseEvent {

	MouseEvent() {
		event = -1;
		buttonState = 0;
	}
	cv::Point pt;
	int event;
	int buttonState;

};

static void onMouse( int event, int x, int y, int flags, void* userdata ) {
	if( userdata ) {
		MouseEvent* data = (MouseEvent*) userdata;
		data->event = event;
		data->pt = cv::Point( x, y );
		data->buttonState = flags;
	}
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

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	;
};

class TrainObjectFromData {
public:

	TrainObjectFromData( int argc, char** argv )
			: viewer_( &slam_ ) {

		imageAllocator_ = boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator >( new MultiResolutionSurfelMap::ImagePreAllocator() );

		ChessboardInfo cbInfo;

		po::options_description desc( "Allowed options" );
		desc.add_options()( "help,h", "help" )( "object,o", po::value< std::string >( &object_name_ )->default_value( "object" ), "object name" )( "inputpath,i",
				po::value< std::string >( &input_path_ )->default_value( "." ), "input path" )( "startframe,s", po::value< int >( &start_frame_ )->default_value( 0 ), "start frame" )( "endframe,e",
				po::value< int >( &end_frame_ )->default_value( 1000 ), "end frame" )( "closeloop,c", "close loop between first and last key frame" )( "checkerboard,b", "use checkerboard" )(
				"boardwidth", po::value< int >( &cbInfo.size.width )->default_value( 5 ), "checkerboard corners along width" )( "boardheight",
				po::value< int >( &cbInfo.size.height )->default_value( 4 ), "checkerboard corners along height" )( "boardsize", po::value< double >( &cbInfo.squareSize )->default_value( 0.3 ),
				"checkerboard square size in m" )( "minheight", po::value< double >( &minHeight_ )->default_value( 0.02 ), "minimum height in m" )( "maxheight",
				po::value< double >( &maxHeight_ )->default_value( 1.0 ), "maximum height in m" )( "maxres", po::value< double >( &min_resolution_ )->default_value( 0.0125 ),
				"maximum resolution in m" )( "maxrange", po::value< double >( &max_range_ )->default_value( 30.0 ), "maximum range in m" );

		po::variables_map vm;
		po::store( po::parse_command_line( argc, argv, desc ), vm );
		po::notify( vm );

		if( vm.count( "help" ) || vm.count( "h" ) ) {
			std::cout << desc << "\n";
			exit( 0 );
		}

		if( vm.count( "checkerboard" ) || vm.count( "b" ) )
			use_cb_ = true;
		else
			use_cb_ = false;

		if( vm.count( "closeloop" ) || vm.count( "c" ) )
			closeLoop_ = true;
		else
			closeLoop_ = false;

		if( use_cb_ ) {

			std::cout << "using board: " << cbInfo.size.width << "x" << cbInfo.size.height << ", square size " << cbInfo.squareSize << "\n";

			cbInfo.initCorners();

			chessboard_ = cbInfo;

		}

	}

	bool selectConvexHull( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGBA >::Ptr& cloud, Eigen::Matrix4d& referenceTransform ) {

		// let user select convex hull points in the images
		std::cout << "select convex hull in the image\n";
		std::cout << "left click: add point, right click: finish selection\n";

		cv::Mat img_cv;
		cv::cvtColor( img_rgb, img_cv, cv::COLOR_BGR2GRAY );

		cv::Mat cameraMatrix, distortionCoeffs;
		getCameraCalibration( cameraMatrix, distortionCoeffs );

		// wait for user input
		// left click: save clicked point in reference frame
		// right/middle click: stop selection

		bool stopSelection = false;
		while( !stopSelection ) {

			// project selected points into image
			cv::Mat img_viz = img_rgb.clone();
			if( convexHull_.size() > 0 ) {

				std::vector< cv::Point3f > convexHullCamera( convexHull_.size() );
				for( unsigned int j = 0; j < convexHull_.size(); j++ ) {

					// transform point from reference frame to camera frame
					Eigen::Vector4d p;
					p[ 0 ] = convexHull_[ j ]( 0 );
					p[ 1 ] = convexHull_[ j ]( 1 );
					p[ 2 ] = convexHull_[ j ]( 2 );
					p[ 3 ] = 1;

					p = ( referenceTransform.inverse() * p ).eval();
					convexHullCamera[ j ].x = p[ 0 ];
					convexHullCamera[ j ].y = p[ 1 ];
					convexHullCamera[ j ].z = p[ 2 ];

				}

				std::vector< cv::Point2f > imagePoints( convexHullCamera.size() );
				cv::Mat rot( 3, 1, CV_64FC1, 0.f );
				cv::Mat trans( 3, 1, CV_64FC1, 0.f );
				cv::projectPoints( cv::Mat( convexHullCamera ), rot, trans, cameraMatrix, distortionCoeffs, imagePoints );

				for( unsigned int j = 0; j < imagePoints.size(); j++ ) {

					if( imagePoints[ j ].x < 0 || imagePoints[ j ].x > img_cv.cols - 1 )
						continue;

					if( imagePoints[ j ].y < 0 || imagePoints[ j ].y > img_cv.rows - 1 )
						continue;

					cv::Scalar c( 0.f, 0.f, 255.f );
					cv::circle( img_viz, imagePoints[ j ], 10, c, -1 );

				}
			}

			int displayHeight = 240;
			double imgScaleFactor = ( (float) displayHeight ) / ( (float) img_viz.rows );
			if( img_viz.rows != displayHeight ) {
				cv::Mat tmp;
				cv::resize( img_viz, tmp, cv::Size(), imgScaleFactor, imgScaleFactor, cv::INTER_LINEAR );
				tmp.copyTo( img_viz );
			}
			cv::imshow( "Select Convex Hull", img_viz );

			MouseEvent mouse;
			cv::setMouseCallback( "Select Convex Hull", onMouse, &mouse );
			cv::waitKey( 10 );

			if( mouse.event == CV_EVENT_LBUTTONDOWN ) {

				// find corresponding 3D position in point cloud
				float img2cloudScale = ( (float) cloud->height ) / ( (float) displayHeight );
				unsigned int idx = round( img2cloudScale * ( (float) mouse.pt.y ) ) * cloud->width + round( img2cloudScale * ( (float) mouse.pt.x ) );
				if( idx < cloud->points.size() && !isnan( cloud->points[ idx ].x ) ) {

					//  transform point to reference frame
					Eigen::Vector4d p;
					p[ 0 ] = cloud->points[ idx ].x;
					p[ 1 ] = cloud->points[ idx ].y;
					p[ 2 ] = cloud->points[ idx ].z;
					p[ 3 ] = 1;

					p = ( referenceTransform * p ).eval();

					convexHull_.push_back( p.cast< float >().block< 3, 1 >( 0, 0 ) );
				}

			}
			else if( mouse.event == CV_EVENT_RBUTTONDOWN ) {
				stopSelection = true;
			}
			else if( mouse.event == CV_EVENT_MBUTTONDOWN ) {
				stopSelection = true;
			}
		}

		if( convexHull_.size() < 3 ) {
			std::cout << "convex hull requires more than 3 points\n";
			return false;
		}
		else {

			// project selected points on common plane

			Eigen::Vector4f plane_parameters;

			// Use Least-Squares to fit the plane through all the given sample points and find out its coefficients
			EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
			Eigen::Vector4f xyz_centroid;

			pcl::PointCloud< pcl::PointXYZ >::Ptr selectedPoints( new pcl::PointCloud< pcl::PointXYZ >() );
			for( unsigned int i = 0; i < convexHull_.size(); i++ ) {
				pcl::PointXYZ p;
				p.x = convexHull_[ i ]( 0 );
				p.y = convexHull_[ i ]( 1 );
				p.z = convexHull_[ i ]( 2 );
				selectedPoints->points.push_back( p );
			}

			// Estimate the XYZ centroid
			pcl::compute3DCentroid( *selectedPoints, xyz_centroid );
			xyz_centroid[ 3 ] = 0;

			// Compute the 3x3 covariance matrix
			pcl::computeCovarianceMatrix( *selectedPoints, xyz_centroid, covariance_matrix );

			// Compute the model coefficients
			EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
			EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
			pcl::eigen33( covariance_matrix, eigen_vectors, eigen_values );

			// remove components orthogonal to the plane..
			for( unsigned int i = 0; i < convexHull_.size(); i++ ) {

				Eigen::Vector3f p = convexHull_[ i ];

				float l = p.dot( eigen_vectors.block< 3, 1 >( 0, 0 ) ) - xyz_centroid.block< 3, 1 >( 0, 0 ).dot( eigen_vectors.block< 3, 1 >( 0, 0 ) );

				p -= l * eigen_vectors.block< 3, 1 >( 0, 0 );

				convexHull_[ i ]( 0 ) = p( 0 );
				convexHull_[ i ]( 1 ) = p( 1 );
				convexHull_[ i ]( 2 ) = p( 2 );

			}

		}

		return true;

	}

	void train() {

		if( !use_cb_ ) {

			// parse groundtruth.txt
			std::ifstream gtFile( ( input_path_ + std::string( "/groundtruth.txt" ) ).c_str() );

			while( gtFile.good() ) {

				// read in line
				char lineCStr[ 1024 ];
				gtFile.getline( lineCStr, 1024, '\n' );

				std::string lineStr( lineCStr );

				// split line at blanks
				std::vector< std::string > entryStrs;
				boost::split( entryStrs, lineStr, boost::is_any_of( "\t " ) );

				if( entryStrs.size() != 8 )
					continue;

				std::stringstream sstr;
				sstr << entryStrs[ 0 ];
				double stamp = 0.0;
				sstr >> stamp;

				double tx, ty, tz, qx, qy, qz, qw;
				sstr.clear();
				sstr << entryStrs[ 1 ];
				sstr >> tx;
				sstr.clear();
				sstr << entryStrs[ 2 ];
				sstr >> ty;
				sstr.clear();
				sstr << entryStrs[ 3 ];
				sstr >> tz;
				sstr.clear();
				sstr << entryStrs[ 4 ];
				sstr >> qx;
				sstr.clear();
				sstr << entryStrs[ 5 ];
				sstr >> qy;
				sstr.clear();
				sstr << entryStrs[ 6 ];
				sstr >> qz;
				sstr.clear();
				sstr << entryStrs[ 7 ];
				sstr >> qw;

				Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
				transform.block< 3, 3 >( 0, 0 ) = Eigen::Quaterniond( qw, qx, qy, qz ).matrix();
				transform.block< 3, 1 >( 0, 3 ) = Eigen::Vector3d( tx, ty, tz );

				groundTruth_[ stamp ] = transform;

			}

		}

		// parse associations.txt
		std::ifstream assocFile( ( input_path_ + std::string( "/associations.txt" ) ).c_str() );

		int count = -1;

		std::vector< PoseInfo, Eigen::aligned_allocator< PoseInfo > > trajectoryEstimate;

		unsigned int frameIdx = 0;

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

				// load images
				cv::Mat depthImg = cv::imread( input_path_ + "/" + entryStrs[ 1 ], CV_LOAD_IMAGE_ANYDEPTH );
				cv::Mat rgbImg = cv::imread( input_path_ + "/" + entryStrs[ 3 ], CV_LOAD_IMAGE_ANYCOLOR );

				// extract point cloud from image pair
				pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud( new pcl::PointCloud< pcl::PointXYZRGBA >() );
				imagesToPointCloud( depthImg, rgbImg, entryStrs[ 0 ], cloud );

				std::stringstream sstr;
				sstr << entryStrs[ 0 ];
				double stamp = 0.0;
				sstr >> stamp;

				processFrame( cloud, rgbImg, frameIdx, stamp );

				if( frameIdx == end_frame_ )
					break;

				if( graphChanged_ || viewer_.forceRedraw ) {
					viewer_.visualizeSLAMGraph();
					viewer_.forceRedraw = false;
					graphChanged_ = false;
				}

				if( !viewer_.is_running )
					exit( -1 );

				viewer_.spinOnce();
				usleep( 1000 );

				frameIdx++;

			}

		}

	}

	void processFrame( pcl::PointCloud< pcl::PointXYZRGBA >::Ptr& cloud, const cv::Mat& img_rgb, unsigned int frameIdx, double stamp ) {

		float register_start_resolution = min_resolution_;
		const float register_stop_resolution = 32.f * min_resolution_;

		if( frameIdx >= start_frame_ && frameIdx <= end_frame_ ) {

			std::cout << "processing frame " << frameIdx << "\n";

			// build map in reference frame
			bool isFirstFrame = false;

			if( !use_cb_ ) {

				// use available ground truth for initialization of the reference frame, if cb not used
				if( frameIdx == start_frame_ ) {

					referenceTransform_.setIdentity();

					GTMap::iterator it = groundTruth_.upper_bound( stamp );
					if( it != groundTruth_.end() ) {
						referenceTransform_ = it->second;
						std::cout << "initialized reference frame from ground truth\n";
						std::cout << referenceTransform_;
					}

					isFirstFrame = true;

				}

			}
			else {

				if( !chessboard_.trackInitialized ) {

					// detect cb in image..

					cv::Mat img_cv;
					cv::cvtColor( img_rgb, img_cv, cv::COLOR_BGR2GRAY );

					cv::Mat cameraMatrix, distortionCoeffs;
					getCameraCalibration( cameraMatrix, distortionCoeffs );

					bool boardFound = false;

					std::vector< cv::Point2f > foundBoardCorners;
					boardFound = findChessboardCorners( img_cv, chessboard_.size, foundBoardCorners, cv::CALIB_CB_ADAPTIVE_THRESH );
					if( boardFound ) {

						std::cout << "found board\n";
						cv::cornerSubPix( img_cv, foundBoardCorners, cv::Size( 5, 5 ), cv::Size( -1, -1 ), cv::TermCriteria( CV_TERMCRIT_ITER, 20, 1e-2 ) );

						// extract chessboard pose, use last pose as initial guess
						cv::Mat last_rotation( 3, 1, CV_64FC1 );
						chessboard_.rotation.copyTo( last_rotation );
						cv::Mat last_translation( 3, 1, CV_64FC1 );
						chessboard_.translation.copyTo( last_translation );

						cv::solvePnP( cv::Mat( chessboard_.corners ), cv::Mat( foundBoardCorners ), cameraMatrix, distortionCoeffs, chessboard_.rotation, chessboard_.translation, false );

						chessboard_.trackInitialized = true;

						cv::Mat R( 3, 3, CV_64FC1 );
						cv::Rodrigues( chessboard_.rotation, R );

						referenceTransform_ = Eigen::Matrix4d::Identity();
						for( int y = 0; y < 3; y++ ) {
							for( int x = 0; x < 3; x++ )
								referenceTransform_( y, x ) = R.at< double >( y, x );
							referenceTransform_( y, 3 ) = chessboard_.translation.at< double >( y, 0 );
						}

						Eigen::Matrix4d fixTransform = Eigen::Matrix4d::Identity();
						fixTransform.block< 3, 3 >( 0, 0 ) = Eigen::AngleAxisd( M_PI, Eigen::Vector3d( 1.0, 0.0, 0.0 ) ).matrix();

						referenceTransform_ = referenceTransform_.inverse().eval();
						referenceTransform_ = ( fixTransform * referenceTransform_ ).eval();

						isFirstFrame = true;

					}

				}

			}

			if( isFirstFrame ) {

				selectConvexHull( img_rgb, cloud, referenceTransform_ );

			}

			if( !use_cb_ || chessboard_.trackInitialized ) {

				int numEdges = slam_.optimizer_->edges().size();
				int numVertices = slam_.optimizer_->vertices().size();
				int referenceID = slam_.referenceKeyFrameId_;

				bool retVal = slam_.addImage( img_rgb, cloud, register_start_resolution, register_stop_resolution, min_resolution_, true );

				slam_.refineWorstEdges( 0.05, register_start_resolution, register_stop_resolution );

				if( slam_.optimizer_->vertices().size() != numVertices || slam_.optimizer_->edges().size() != numEdges || slam_.referenceKeyFrameId_ != referenceID )
					graphChanged_ = true;

				if( slam_.optimizer_->vertices().size() > 0 ) {

					g2o::VertexSE3* v_ref = dynamic_cast< g2o::VertexSE3* >( slam_.optimizer_->vertex( slam_.referenceKeyFrameId_ ) );
					Eigen::Matrix4d pose_ref = v_ref->estimate().matrix();

					viewer_.displayPose( pose_ref * slam_.lastTransform_ );

				}

			}

		}

		if( frameIdx == end_frame_ ) {

			Eigen::Matrix4d transformGuess;
			transformGuess.setIdentity();
			if( closeLoop_ ) {
				std::cout << "connecting last and first frame\n";
				slam_.addEdge( 0, slam_.keyFrames_.size() - 1, transformGuess, register_start_resolution, register_stop_resolution, false );
			}

			if( slam_.optimizer_->vertices().size() >= 3 ) {

				for( int i = 0; i < 3; i++ ) {

					slam_.connectClosePoses( register_start_resolution, register_stop_resolution );

					// optimize slam graph
					std::cout << "optimizing...\n";
					slam_.optimizer_->initializeOptimization();
					slam_.optimizer_->optimize( 1 );
					slam_.optimizer_->computeActiveErrors();
					std::cout << slam_.optimizer_->vertices().size() << " nodes, " << slam_.optimizer_->edges().size() << " edges, " << "chi2: " << slam_.optimizer_->chi2() << "\n";

				}
			}

			for( int i = 0; i < 3; i++ ) {
				slam_.connectClosePoses( register_start_resolution, register_stop_resolution );
				slam_.refine( 1, 20, register_start_resolution, register_stop_resolution );
				viewer_.visualizeSLAMGraph();
				viewer_.spinOnce();
				usleep( 10 );
			}

			boost::shared_ptr< MultiResolutionSurfelMap > graphMap = slam_.getMapInConvexHull( referenceTransform_, min_resolution_, minHeight_, maxHeight_, convexHull_ );

			for( unsigned int i = 0; i < slam_.keyFrames_.size(); i++ ) {
				char str[ 255 ];
				sprintf( str, "map%i", i );
				viewer_.viewer->removePointCloud( str );
			}

			graphMap->save( input_path_ + "/" + object_name_ + ".map" );

			while( viewer_.is_running ) {

				pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud2 = pcl::PointCloud< pcl::PointXYZRGBA >::Ptr( new pcl::PointCloud< pcl::PointXYZRGBA >() );
				graphMap->visualize3DColorDistribution( cloud2, viewer_.selectedDepth, viewer_.selectedViewDir, false );
				viewer_.displayPointCloud( "finalmap", cloud2 );

				viewer_.spinOnce();
				usleep( 2000 );
			}

		}

	}

public:

	std::string object_name_;
	std::string input_path_;
	double minHeight_, maxHeight_;

	std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > convexHull_;

	Eigen::Matrix4d referenceTransform_;
	SLAM slam_;

	boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator > imageAllocator_;

	double min_resolution_, max_range_;
	int start_frame_, end_frame_;

	bool use_cb_; // use checkerboard instead of mocap for first frame
	ChessboardInfo chessboard_;

	bool closeLoop_;

	bool graphChanged_;

	ViewerSLAM viewer_;

	typedef std::map< double, Eigen::Matrix4d, std::less< double >, Eigen::aligned_allocator< std::pair< const double, Eigen::Matrix4d > > > GTMap;
	GTMap groundTruth_;

};

int main( int argc, char** argv ) {

	TrainObjectFromData tofd( argc, argv );

	tofd.train();

	while( tofd.viewer_.is_running ) {

		if( tofd.viewer_.forceRedraw ) {
			tofd.viewer_.visualizeSLAMGraph();
			tofd.viewer_.forceRedraw = false;
		}

		tofd.viewer_.spinOnce();
		usleep( 1000 );
	}

	return 0;
}

