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

#include <mrsmap/slam/slam.h>

#include <pcl/registration/transforms.h>

#include <mrsmap/registration/multiresolution_csurfel_registration.h>

#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/surface/convex_hull.h>

#include <g2o/core/optimization_algorithm_levenberg.h>

using namespace mrsmap;

#define GRADIENT_ITS 100
#define NEWTON_FEAT_ITS 0
#define NEWTON_ITS 5

#define LOG_LIKELIHOOD_ADD_THRESHOLD -150000

#define REGISTER_TWICE 0

SLAM::SLAM() {

	srand( time( NULL ) );

	imageAllocator_ = boost::shared_ptr< MultiResolutionSurfelMap::ImagePreAllocator >( new MultiResolutionSurfelMap::ImagePreAllocator() );
	treeNodeAllocator_ = boost::shared_ptr< spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue > >(
			new spatialaggregate::OcTreeNodeDynamicAllocator< float, MultiResolutionSurfelMap::NodeValue >( 1000 ) );

	referenceKeyFrameId_ = 0;
	lastTransform_.setIdentity();
	lastFrameTransform_.setIdentity();

	// allocating the optimizer
	optimizer_ = new g2o::SparseOptimizer();
	optimizer_->setVerbose( true );
	SlamLinearSolver* linearSolver = new SlamLinearSolver();
	linearSolver->setBlockOrdering( false );
	SlamBlockSolver* solver = new SlamBlockSolver( linearSolver );

	g2o::OptimizationAlgorithmLevenberg* solverLevenberg = new g2o::OptimizationAlgorithmLevenberg( solver );

	optimizer_->setAlgorithm( solverLevenberg );

}

SLAM::~SLAM() {

	delete optimizer_;

}

unsigned int SLAM::addKeyFrame( unsigned int kf_prev_id, boost::shared_ptr< KeyFrame >& keyFrame, const Eigen::Matrix4d& transform ) {

	keyFrame->nodeId_ = optimizer_->vertices().size();

	// anchor first frame at origin
	if( keyFrames_.empty() ) {

		g2o::VertexSE3* v = new g2o::VertexSE3();
		v->setId( keyFrame->nodeId_ );
		v->setEstimate( g2o::SE3Quat() );
		v->setFixed( true );
		optimizer_->addVertex( v );
		keyFrames_.push_back( keyFrame );
		keyFrameNodeMap_[ keyFrame->nodeId_ ] = keyFrame;

	}
	else {

		g2o::SE3Quat measurement_mean( Eigen::Quaterniond( transform.block< 3, 3 >( 0, 0 ) ), transform.block< 3, 1 >( 0, 3 ) );

		g2o::VertexSE3* v_prev = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[ kf_prev_id ]->nodeId_ ) );

		// create vertex in slam graph for new key frame
		g2o::VertexSE3* v = new g2o::VertexSE3();
		v->setId( keyFrame->nodeId_ );
		v->setEstimate( v_prev->estimate() * measurement_mean );
		optimizer_->addVertex( v );
		keyFrames_.push_back( keyFrame );
		keyFrameNodeMap_[ keyFrame->nodeId_ ] = keyFrame;

	}

	return keyFrames_.size() - 1;

}

unsigned int SLAM::addIntermediateFrame( unsigned int kf_ref_id, boost::shared_ptr< MultiResolutionSurfelMap >& currFrame, const Eigen::Matrix4d& transform ) {

	unsigned int v_id = optimizer_->vertices().size();

	// anchor first frame at origin
	if( keyFrames_.empty() ) {

		std::cerr << "ERROR: first frame should not be intermediate frame!\n";
		exit( -1 );
		return 0;

	}
	else {

		g2o::SE3Quat measurement_mean( Eigen::Quaterniond( transform.block< 3, 3 >( 0, 0 ) ), transform.block< 3, 1 >( 0, 3 ) );

		g2o::VertexSE3* v_prev = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[ kf_ref_id ]->nodeId_ ) );

		// create vertex in slam graph for new key frame
		g2o::VertexSE3* v = new g2o::VertexSE3();
		v->setId( v_id );
		v->setEstimate( v_prev->estimate() * measurement_mean );
		optimizer_->addVertex( v );

	}

	return v_id;

}

bool SLAM::addEdge( unsigned int v1_id, unsigned int v2_id, float register_start_resolution, float register_stop_resolution, bool checkMatchingLikelihood ) {

	g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );
	g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

	// diff transform from v2 to v1
	Eigen::Matrix4d diffTransform = ( v1->estimate().inverse() * v2->estimate() ).matrix();

	// add edge to graph
	return addEdge( v1_id, v2_id, diffTransform, register_start_resolution, register_stop_resolution, checkMatchingLikelihood );

}

bool SLAM::addEdge( unsigned int v1_id, unsigned int v2_id, const Eigen::Matrix4d& transformGuess, float register_start_resolution, float register_stop_resolution, bool checkMatchingLikelihood ) {

	g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );
	g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

	Eigen::Matrix4d transform = transformGuess;

	// register maps with pose guess from graph
	Eigen::Matrix< double, 6, 6 > poseCov;

	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr corrSrc;
	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr corrTgt;
	MultiResolutionColorSurfelRegistration reg;
	bool retVal = reg.estimateTransformation( *( keyFrameNodeMap_[ v1_id ]->map_ ), *( keyFrameNodeMap_[ v2_id ]->map_ ), transform, register_start_resolution, register_stop_resolution, corrSrc,
			corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );
	if( REGISTER_TWICE )
		retVal = reg.estimateTransformation( *( keyFrameNodeMap_[ v1_id ]->map_ ), *( keyFrameNodeMap_[ v2_id ]->map_ ), transform, register_start_resolution, register_stop_resolution, corrSrc,
				corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );
	if( !retVal )
		return false;

	Eigen::Matrix4d transforminv = transform.inverse();
	double logLikelihood1 = reg.matchLogLikelihood( *( keyFrameNodeMap_[ v1_id ]->map_ ), *( keyFrameNodeMap_[ v2_id ]->map_ ), transforminv );
	std::cout << "new edge likelihood1: " << logLikelihood1 << "\n";

	double logLikelihood2 = reg.matchLogLikelihood( *( keyFrameNodeMap_[ v2_id ]->map_ ), *( keyFrameNodeMap_[ v1_id ]->map_ ), transform );
	std::cout << "new edge likelihood2: " << logLikelihood2 << "\n";

	if( checkMatchingLikelihood ) {
		double baseLogLikelihood1 = keyFrameNodeMap_[ v1_id ]->sumLogLikelihood_ / keyFrameNodeMap_[ v1_id ]->numEdges_;
		double baseLogLikelihood2 = keyFrameNodeMap_[ v2_id ]->sumLogLikelihood_ / keyFrameNodeMap_[ v2_id ]->numEdges_;
		std::cout << "key frame1 base log likelihood is " << baseLogLikelihood1 << "\n";
		std::cout << "key frame2 base log likelihood is " << baseLogLikelihood2 << "\n";

		if( logLikelihood1 < baseLogLikelihood1 + LOG_LIKELIHOOD_ADD_THRESHOLD || logLikelihood2 < baseLogLikelihood2 + LOG_LIKELIHOOD_ADD_THRESHOLD ) {
			std::cout << "============= BAD MATCHING LIKELIHOOD ============\n";
			return false;
		}
	}

	retVal = reg.estimatePoseCovariance( poseCov, *( keyFrameNodeMap_[ v1_id ]->map_ ), *( keyFrameNodeMap_[ v2_id ]->map_ ), transform, register_start_resolution, register_stop_resolution );

	if( !retVal )
		return false;

	// add edge to graph
	return addEdge( v1_id, v2_id, transform, poseCov );

}

// returns true, iff node could be added to the cloud
bool SLAM::addEdge( unsigned int v1_id, unsigned int v2_id, const Eigen::Matrix4d& transform, const Eigen::Matrix< double, 6, 6 >& covariance ) {

	unsigned int edges = optimizer_->edges().size();

	g2o::SE3Quat measurement_mean( Eigen::Quaterniond( transform.block< 3, 3 >( 0, 0 ) ), transform.block< 3, 1 >( 0, 3 ) );
	Eigen::Matrix< double, 6, 6 > measurement_information = covariance.inverse();

	g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );
	g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

	// create edge between new key frame and previous key frame with the estimated transformation
	g2o::EdgeSE3* edge = new g2o::EdgeSE3();
	edge->vertices()[ 0 ] = v1;
	edge->vertices()[ 1 ] = v2;
	edge->setMeasurement( measurement_mean );
	edge->setInformation( measurement_information );

	return optimizer_->addEdge( edge );

}

bool SLAM::poseIsClose( const Eigen::Matrix4d& transform ) {

	double angle = Eigen::AngleAxisd( transform.block< 3, 3 >( 0, 0 ) ).angle();
	double dist = transform.block< 3, 1 >( 0, 3 ).norm();

	return fabsf( angle ) < 0.2f && dist < 0.3f;
}

bool SLAM::poseIsFar( const Eigen::Matrix4d& transform ) {

	double angle = Eigen::AngleAxisd( transform.block< 3, 3 >( 0, 0 ) ).angle();
	double dist = transform.block< 3, 1 >( 0, 3 ).norm();

	return fabsf( angle ) > 0.4f || dist > 0.7f;
}

bool SLAM::addImage( const cv::Mat& img_rgb, const pcl::PointCloud< pcl::PointXYZRGBA >::ConstPtr& pointCloudIn, float startResolution, float stopResolution, float minResolution, bool storeCloud ) {

	const int numPoints = pointCloudIn->points.size();

	std::vector< int > indices( numPoints );
	for( int i = 0; i < numPoints; i++ )
		indices[ i ] = i;

	const float register_start_resolution = startResolution;
	const float register_stop_resolution = stopResolution;

	const float min_resolution = minResolution;
	const float max_radius = 30.f;

	// slam graph: list of key frames
	// match current frame to last key frame
	// create new key frame after some delta in translation or rotation

	boost::shared_ptr< MultiResolutionSurfelMap > target = boost::shared_ptr< MultiResolutionSurfelMap >( new MultiResolutionSurfelMap( min_resolution, max_radius ) );
	Eigen::Matrix4d incTransform = lastTransform_;

	// add points to local map
	target->imageAllocator_ = imageAllocator_;
	target->addImage( *pointCloudIn );
	std::vector< int > imageBorderIndices;
	target->findVirtualBorderPoints( *pointCloudIn, imageBorderIndices );
	target->markNoUpdateAtPoints( *pointCloudIn, imageBorderIndices );
	target->evaluateSurfels();
	target->octree_->root_->establishNeighbors();
	target->buildShapeTextureFeatures();

	bool generateKeyFrame = false;

	Eigen::Matrix4d currFrameTransform = Eigen::Matrix4d::Identity();

	if( keyFrames_.empty() ) {

		generateKeyFrame = true;

	}
	else {

		pcl::PointCloud< pcl::PointXYZRGBA >::Ptr corrSrc;
		pcl::PointCloud< pcl::PointXYZRGBA >::Ptr corrTgt;
		MultiResolutionColorSurfelRegistration reg;
		bool retVal = reg.estimateTransformation( *( keyFrames_[ referenceKeyFrameId_ ]->map_ ), *target, incTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt,
				GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );
		if( REGISTER_TWICE )
			retVal = reg.estimateTransformation( *( keyFrames_[ referenceKeyFrameId_ ]->map_ ), *target, incTransform, register_start_resolution, register_stop_resolution, corrSrc, corrTgt,
					GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );

		if( retVal ) {
			lastTransform_ = incTransform;

			// check for sufficient pose delta to generate a new key frame

			if( !poseIsClose( lastTransform_ ) ) {
				generateKeyFrame = true;
			}
		}
		else {
			std::cout << "SLAM: lost track in current frame\n";
			exit( -1 );
			return false;
		}

	}

	if( generateKeyFrame ) {

		boost::shared_ptr< KeyFrame > keyFrame = boost::shared_ptr< KeyFrame >( new KeyFrame() );

		if( storeCloud ) {
			keyFrame->cloud_ = pointCloudIn;
			keyFrame->img_rgb = img_rgb;
		}

		keyFrame->map_ = target;

		// evaluate pose covariance between keyframes..
		Eigen::Matrix< double, 6, 6 > poseCov;

		if( !keyFrames_.empty() ) {

			pcl::PointCloud< pcl::PointXYZRGBA >::Ptr corrSrc;
			pcl::PointCloud< pcl::PointXYZRGBA >::Ptr corrTgt;
			MultiResolutionColorSurfelRegistration reg;
			reg.estimatePoseCovariance( poseCov, *( keyFrames_[ referenceKeyFrameId_ ]->map_ ), *( keyFrame->map_ ), lastTransform_, register_start_resolution, register_stop_resolution );

			Eigen::Matrix4d Tinv = lastTransform_.inverse();
			double logLikelihood1 = reg.matchLogLikelihood( *( keyFrames_[ referenceKeyFrameId_ ]->map_ ), *( keyFrame->map_ ), Tinv );
			double logLikelihood2 = reg.matchLogLikelihood( *( keyFrame->map_ ), *( keyFrames_[ referenceKeyFrameId_ ]->map_ ), lastTransform_ );
			std::cout << "new key frame first edge log likelihood: " << logLikelihood2 << "\n";

			// store average log likelihood information of edges to the keyframes
			keyFrames_[ referenceKeyFrameId_ ]->sumLogLikelihood_ += logLikelihood1;
			keyFrames_[ referenceKeyFrameId_ ]->numEdges_ += 1.0;

			keyFrame->sumLogLikelihood_ += logLikelihood2;
			keyFrame->numEdges_ += 1.0;
		}
		else
			poseCov.setZero();

		// extend slam graph with vertex for new key frame and with one edge towards the last keyframe..
		unsigned int keyFrameId = addKeyFrame( referenceKeyFrameId_, keyFrame, lastTransform_ );
		if( optimizer_->vertices().size() > 1 ) {
			if( !addEdge( keyFrames_[ referenceKeyFrameId_ ]->nodeId_, keyFrames_[ keyFrameId ]->nodeId_, lastTransform_, poseCov ) ) {
				std::cout << "WARNING: new key frame not connected to graph!\n";
				assert( false );
			}
		}

		assert( optimizer_->vertices().size() == keyFrames_.size() );

	}

	// try to match between older key frames (that are close in optimized pose)
	connectClosePoses( startResolution, stopResolution, true );

	if( optimizer_->vertices().size() >= 3 ) {
		// optimize slam graph
		std::cout << "optimizing...\n";
		optimizer_->initializeOptimization();
		optimizer_->optimize( 1 );
		optimizer_->computeActiveErrors();
		std::cout << optimizer_->vertices().size() << " nodes, " << optimizer_->edges().size() << " edges, " << "chi2: " << optimizer_->chi2() << "\n";
	}

	// get estimated transform in map frame
	unsigned int oldReferenceId_ = referenceKeyFrameId_;
	g2o::VertexSE3* v_ref_old = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[ oldReferenceId_ ]->nodeId_ ) );
	Eigen::Matrix4d pose_ref_old = v_ref_old->estimate().matrix();
	Eigen::Matrix4d tracked_pose = pose_ref_old * lastTransform_;

	unsigned int bestId = optimizer_->vertices().size() - 1;

	// select closest key frame to current camera pose for further tracking
	// in this way, we do not create unnecessary key frames..
	float bestAngle = std::numeric_limits< float >::max();
	float bestDist = std::numeric_limits< float >::max();
	for( unsigned int kf_id = 0; kf_id < keyFrames_.size(); kf_id++ ) {

		g2o::VertexSE3* v = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[ kf_id ]->nodeId_ ) );

		Eigen::Matrix4d v_pose = v->estimate().matrix();

		Eigen::Matrix4d diffTransform = v_pose.inverse() * tracked_pose;

		double angle = Eigen::AngleAxisd( diffTransform.block< 3, 3 >( 0, 0 ) ).angle();
		double dist = diffTransform.block< 3, 1 >( 0, 3 ).norm();

		if( poseIsClose( diffTransform ) && fabsf( angle ) < bestAngle && dist < bestDist ) {
			bestAngle = angle;
			bestDist = dist;
			bestId = kf_id;
		}

	}

	// try to add new edge between the two reference
	// if not possible, we keep the old reference frame such that a new key frame will added later that connects the two reference frames
	bool switchReferenceID = true;
	g2o::VertexSE3* v_ref = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[ referenceKeyFrameId_ ]->nodeId_ ) );

	if( switchReferenceID ) {
		referenceKeyFrameId_ = bestId;
	}

	// set lastTransform_ to pose wrt reference key frame
	v_ref = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( keyFrames_[ referenceKeyFrameId_ ]->nodeId_ ) );
	Eigen::Matrix4d pose_ref = v_ref->estimate().matrix();
	lastTransform_ = pose_ref.inverse() * tracked_pose;

	return true;

}

void SLAM::connectClosePoses( float register_start_resolution, float register_stop_resolution, bool random ) {

	// random == true: randomly check only one vertex, the closer, the more probable the check
	if( random ) {

		const double sigma2_dist = 0.7 * 0.7;
		const double sigma2_angle = 0.3 * 0.3;

		for( unsigned int kf1_id = referenceKeyFrameId_; kf1_id <= referenceKeyFrameId_; kf1_id++ ) {

			unsigned int v1_id = keyFrames_[ kf1_id ]->nodeId_;
			g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );

			std::vector< int > vertices;
			std::vector< double > probs;
			double sumProbs = 0.0;

			for( unsigned int kf2_id = 0; kf2_id < kf1_id; kf2_id++ ) {

				unsigned int v2_id = keyFrames_[ kf2_id ]->nodeId_;
				g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

				// check if edge already exists between the vertices
				bool foundEdge = false;
				for( EdgeSet::iterator it = v1->edges().begin(); it != v1->edges().end(); ++it ) {
					g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( *it );
					if( ( edge->vertices()[ 0 ]->id() == v1_id && edge->vertices()[ 1 ]->id() == v2_id ) || ( edge->vertices()[ 0 ]->id() == v2_id && edge->vertices()[ 1 ]->id() == v1_id ) ) {
						foundEdge = true;
						break;
					}
				}
				if( foundEdge )
					continue;

				// diff transform from v2 to v1
				Eigen::Matrix4d diffTransform = ( v1->estimate().inverse() * v2->estimate() ).matrix();

				if( poseIsFar( diffTransform ) )
					continue;

				double angle = Eigen::AngleAxisd( diffTransform.block<3,3>(0,0) ).angle();
				double dist = diffTransform.block<3,1>(0,3).norm();

				// probability of drawing v2 to check for an edge
				double probDist = exp( -0.5 * dist*dist / sigma2_dist );
				double probAngle = exp( -0.5 * angle*angle / sigma2_angle );

				if( probDist > 0.1 && probAngle > 0.1 ) {

					sumProbs += probDist*probAngle;
					probs.push_back( sumProbs );
					vertices.push_back( v2_id );

				}

			}

			if( probs.size() == 0 )
				continue;

			// draw random number in [0,sumProbs]
			double checkProb = (double) rand() / (double) ( RAND_MAX + 1.0 ) * sumProbs;
			for( int i = 0; i < vertices.size(); i++ ) {
				if( checkProb <= probs[ i ] ) {
					int v2_id = vertices[ i ];
					g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );
					Eigen::Matrix4d diffTransform = ( v1->estimate().inverse() * v2->estimate() ).matrix();
					bool retVal = addEdge( v1_id, v2_id, diffTransform, register_start_resolution, register_stop_resolution );
					break;
				}
			}

		}

	}
	else {

		// add all new edges to slam graph
		for( unsigned int kf1_id = 0; kf1_id < keyFrames_.size(); kf1_id++ ) {

			for( unsigned int kf2_id = 0; kf2_id < kf1_id; kf2_id++ ) {

				unsigned int v1_id = keyFrames_[ kf1_id ]->nodeId_;
				unsigned int v2_id = keyFrames_[ kf2_id ]->nodeId_;
				g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );
				g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

				// check if edge already exists between the vertices
				bool foundEdge = false;
				for( EdgeSet::iterator it = v1->edges().begin(); it != v1->edges().end(); ++it ) {
					g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( *it );
					if( ( edge->vertices()[ 0 ]->id() == v1_id && edge->vertices()[ 1 ]->id() == v2_id ) || ( edge->vertices()[ 0 ]->id() == v2_id && edge->vertices()[ 1 ]->id() == v1_id ) ) {
						foundEdge = true;
						break;
					}
				}
				if( foundEdge )
					continue;

				// check if poses close
				// diff transform from v2 to v1
				Eigen::Matrix4d diffTransform = ( v1->estimate().inverse() * v2->estimate() ).matrix();
				if( poseIsFar( diffTransform ) )
					continue;

				bool retVal = addEdge( v1_id, v2_id, diffTransform, register_start_resolution, register_stop_resolution );

			}

		}

	}

}

bool SLAM::refineEdge( g2o::EdgeSE3* edge, float register_start_resolution, float register_stop_resolution ) {

	unsigned int v1_id = edge->vertices()[ 0 ]->id();
	unsigned int v2_id = edge->vertices()[ 1 ]->id();

	g2o::VertexSE3* v1 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v1_id ) );
	g2o::VertexSE3* v2 = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v2_id ) );

	Eigen::Matrix4d diffTransform = ( v1->estimate().inverse() * v2->estimate() ).matrix();

	// register maps with pose guess from graph
	Eigen::Matrix< double, 6, 6 > poseCov;

	if( keyFrameNodeMap_.find( v1_id ) == keyFrameNodeMap_.end() || keyFrameNodeMap_.find( v2_id ) == keyFrameNodeMap_.end() )
		return true; // dont delete this edge!

	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr corrSrc;
	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr corrTgt;
	MultiResolutionColorSurfelRegistration reg;
	bool retVal = reg.estimateTransformation( *( keyFrameNodeMap_[ v1_id ]->map_ ), *( keyFrameNodeMap_[ v2_id ]->map_ ), diffTransform, register_start_resolution, register_stop_resolution, corrSrc,
			corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );
	if( REGISTER_TWICE )
		retVal = reg.estimateTransformation( *( keyFrameNodeMap_[ v1_id ]->map_ ), *( keyFrameNodeMap_[ v2_id ]->map_ ), diffTransform, register_start_resolution, register_stop_resolution, corrSrc,
				corrTgt, GRADIENT_ITS, NEWTON_FEAT_ITS, NEWTON_ITS );
	if( !retVal )
		return false;

	retVal &= reg.estimatePoseCovariance( poseCov, *( keyFrameNodeMap_[ v1_id ]->map_ ), *( keyFrameNodeMap_[ v2_id ]->map_ ), diffTransform, register_start_resolution, register_stop_resolution );

	if( retVal ) {

		g2o::SE3Quat measurement_mean( Eigen::Quaterniond( diffTransform.block< 3, 3 >( 0, 0 ) ), diffTransform.block< 3, 1 >( 0, 3 ) );
		Eigen::Matrix< double, 6, 6 > measurement_information = poseCov.inverse();

		edge->setMeasurement( measurement_mean );
		edge->setInformation( measurement_information );

	}

	return retVal;

}

void SLAM::refine( unsigned int refineIterations, unsigned int optimizeIterations, float register_start_resolution, float register_stop_resolution ) {

	if( optimizer_->vertices().size() >= 3 ) {

		for( unsigned int i = 0; i < refineIterations; i++ ) {

			std::cout << "refining " << i << " / " << refineIterations << "\n";

			// reestimate all edges in the graph from the current pose estimates in the graph
			std::vector< g2o::EdgeSE3* > removeEdges;
			for( EdgeSet::iterator it = optimizer_->edges().begin(); it != optimizer_->edges().end(); ++it ) {

				g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( *it );

				bool retVal = refineEdge( edge, register_start_resolution, register_stop_resolution );

				if( !retVal ) {

					removeEdges.push_back( edge );

				}

			}

			for( unsigned int j = 0; j < removeEdges.size(); j++ )
				optimizer_->removeEdge( removeEdges[ j ] );

			// reoptimize for 10 iterations
			optimizer_->initializeOptimization();
			optimizer_->optimize( 10 );

		}

		// optimize slam graph
		std::cout << "optimizing...\n";
		optimizer_->initializeOptimization();
		optimizer_->optimize( optimizeIterations );
		optimizer_->computeActiveErrors();
		std::cout << optimizer_->vertices().size() << " nodes, " << optimizer_->edges().size() << " edges, " << "chi2: " << optimizer_->chi2() << "\n";
	}

}

void SLAM::refineInConvexHull( unsigned int refineIterations, unsigned int optimizeIterations, float register_start_resolution, float register_stop_resolution, float minResolution,
		const Eigen::Matrix4d& referenceTransform, float minHeight, float maxHeight, std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > convexHull ) {

	const float min_resolution = minResolution;
	const float max_radius = 30.f;

	// restrict all key frames to convex hull from current pose estimate
	pcl::ExtractPolygonalPrismData< pcl::PointXYZRGBA > hull_limiter;

	// extract map and stitched point cloud from selected volume..
	// find convex hull for selected points in reference frame
	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud_selected_points( new pcl::PointCloud< pcl::PointXYZRGBA >() );
	for( unsigned int j = 0; j < convexHull.size(); j++ ) {
		pcl::PointXYZRGBA p;
		p.x = convexHull[ j ]( 0 );
		p.y = convexHull[ j ]( 1 );
		p.z = convexHull[ j ]( 2 );
		cloud_selected_points->points.push_back( p );
	}

	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud_convex_hull( new pcl::PointCloud< pcl::PointXYZRGBA >() );
	pcl::ConvexHull< pcl::PointXYZRGBA > chull;
	chull.setInputCloud( cloud_selected_points );
	chull.reconstruct( *cloud_convex_hull );

	for( unsigned int v_id = 0; v_id < optimizer_->vertices().size(); v_id++ ) {

		g2o::VertexSE3* v = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v_id ) );

		Eigen::Matrix4d v_pose = v->estimate().matrix();

		Eigen::Matrix4d transform = referenceTransform * v_pose;

		pcl::PointCloud< pcl::PointXYZRGBA >::Ptr transformedCloud = pcl::PointCloud< pcl::PointXYZRGBA >::Ptr( new pcl::PointCloud< pcl::PointXYZRGBA >() );
		pcl::transformPointCloud( *( keyFrames_[ v_id ]->cloud_ ), *transformedCloud, transform.cast< float >() );

		transformedCloud->sensor_origin_ = transform.block< 4, 1 >( 0, 3 ).cast< float >();
		transformedCloud->sensor_orientation_ = Eigen::Quaternionf( transform.block< 3, 3 >( 0, 0 ).cast< float >() );

		// get indices in convex hull
		pcl::PointIndices::Ptr object_indices( new pcl::PointIndices() );
		hull_limiter.setInputCloud( transformedCloud );
		hull_limiter.setInputPlanarHull( cloud_convex_hull );
		hull_limiter.setHeightLimits( minHeight, maxHeight );
		hull_limiter.setViewPoint( transformedCloud->sensor_origin_[ 0 ], transformedCloud->sensor_origin_[ 1 ], transformedCloud->sensor_origin_[ 2 ] );
		hull_limiter.segment( *object_indices );

		pcl::PointCloud< pcl::PointXYZRGBA >::Ptr insideCloud = pcl::PointCloud< pcl::PointXYZRGBA >::Ptr( new pcl::PointCloud< pcl::PointXYZRGBA >() );
		*insideCloud = *( keyFrames_[ v_id ]->cloud_ );

		// mark points outside of convex hull nan
		std::vector< int > markNAN( insideCloud->points.size(), 1 );
		for( unsigned int i = 0; i < object_indices->indices.size(); i++ ) {

			markNAN[ object_indices->indices[ i ] ] = 0;

		}

		for( unsigned int i = 0; i < markNAN.size(); i++ ) {

			if( markNAN[ i ] ) {

				insideCloud->points[ i ].x = insideCloud->points[ i ].y = insideCloud->points[ i ].z = std::numeric_limits< float >::quiet_NaN();

			}

		}

		// generate new map for key frame
		boost::shared_ptr< KeyFrame > keyFrame = keyFrames_[ v_id ];
		keyFrame->cloud_ = insideCloud;
		keyFrame->map_ = boost::shared_ptr< MultiResolutionSurfelMap >( new MultiResolutionSurfelMap( min_resolution, max_radius ) );
		keyFrame->map_->imageAllocator_ = imageAllocator_;
		keyFrame->map_->addImage( *insideCloud );
		std::vector< int > imageBorderIndices;
		keyFrame->map_->findVirtualBorderPoints( *insideCloud, imageBorderIndices );
		keyFrame->map_->markNoUpdateAtPoints( *insideCloud, imageBorderIndices );
		keyFrame->map_->evaluateSurfels();
		keyFrame->map_->octree_->root_->establishNeighbors();
		keyFrame->map_->buildShapeTextureFeatures();

	}

	refine( refineIterations, optimizeIterations, register_start_resolution, register_stop_resolution );

}

bool edgeCompareChi( const g2o::HyperGraph::Edge* a, const g2o::HyperGraph::Edge* b ) {
	return ( dynamic_cast< const g2o::EdgeSE3* >( a ) )->chi2() > ( dynamic_cast< const g2o::EdgeSE3* >( b ) )->chi2();
}

void SLAM::refineWorstEdges( float fraction, float register_start_resolution, float register_stop_resolution ) {

	if( optimizer_->vertices().size() >= 3 ) {

		int numRefineEdges = optimizer_->edges().size();
		int refineEdgeIdx = rand() % numRefineEdges;

		// reestimate fraction of edges with worst chi2
		std::vector< g2o::HyperGraph::Edge* > sortedEdges;
		sortedEdges.assign( optimizer_->edges().begin(), optimizer_->edges().end() );
		std::sort( sortedEdges.begin(), sortedEdges.end(), edgeCompareChi );

		std::cout << dynamic_cast< g2o::EdgeSE3* >( *sortedEdges.begin() )->chi2() << " " << dynamic_cast< g2o::EdgeSE3* >( *( sortedEdges.end() - 1 ) )->chi2() << "\n";

		std::vector< g2o::EdgeSE3* > removeEdges;

		g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( sortedEdges[ refineEdgeIdx ] );

		bool retVal = refineEdge( edge, register_start_resolution, register_stop_resolution );

		for( unsigned int j = 0; j < removeEdges.size(); j++ )
			optimizer_->removeEdge( removeEdges[ j ] );

	}

}

void SLAM::dumpError() {

	// dump error of all edges in the slam graph
	std::ofstream outfile( "slam_graph_error.dat" );

	for( EdgeSet::iterator it = optimizer_->edges().begin(); it != optimizer_->edges().end(); ++it ) {
		g2o::EdgeSE3* edge = dynamic_cast< g2o::EdgeSE3* >( *it );

		outfile << edge->chi2() << "\n";

	}
}

boost::shared_ptr< MultiResolutionSurfelMap > SLAM::getMap( const Eigen::Matrix4d& referenceTransform, float minResolution ) {

	const float min_resolution = minResolution;
	const float max_radius = 30.f;

	boost::shared_ptr< MultiResolutionSurfelMap > graphmap = boost::shared_ptr< MultiResolutionSurfelMap >( new MultiResolutionSurfelMap( min_resolution, max_radius ) );
	graphmap->imageAllocator_ = imageAllocator_;

	for( unsigned int v_id = 0; v_id < optimizer_->vertices().size(); v_id++ ) {

		if( keyFrameNodeMap_.find( v_id ) == keyFrameNodeMap_.end() )
			continue;

		g2o::VertexSE3* v = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v_id ) );

		if( v->edges().size() == 0 )
			continue;

		Eigen::Matrix4d v_pose = v->estimate().matrix();

		Eigen::Matrix4d transform = referenceTransform * v_pose;

		pcl::PointCloud< pcl::PointXYZRGBA > transformedCloud;
		pcl::transformPointCloud( *( keyFrameNodeMap_[ v_id ]->cloud_ ), transformedCloud, transform.cast< float >() );

		transformedCloud.sensor_origin_ = transform.block< 4, 1 >( 0, 3 ).cast< float >();
		transformedCloud.sensor_orientation_ = Eigen::Quaternionf( transform.block< 3, 3 >( 0, 0 ).cast< float >() );

		// add keyframe to map
		graphmap->setApplyUpdate( false );
		graphmap->markUpdateImprovedEffViewDistSurfels( transformedCloud.sensor_origin_.block< 3, 1 >( 0, 0 ) );
		std::vector< int > imageBorderIndices;
		graphmap->findVirtualBorderPoints( *( keyFrameNodeMap_[ v_id ]->cloud_ ), imageBorderIndices );
		graphmap->markNoUpdateAtPoints( transformedCloud, imageBorderIndices );
		graphmap->unevaluateSurfels();
		graphmap->addImage( transformedCloud );
		graphmap->clearUpdateSurfelsAtPoints( transformedCloud, imageBorderIndices ); // only new surfels at these points have up_to_date == false !
		graphmap->octree_->root_->establishNeighbors();
		graphmap->clearUnstableSurfels();
		graphmap->setApplyUpdate( true );
		graphmap->evaluateSurfels();
		graphmap->buildShapeTextureFeatures();

	}

	return graphmap;

}

boost::shared_ptr< MultiResolutionSurfelMap > SLAM::getMapInConvexHull( const Eigen::Matrix4d& referenceTransform, float minResolution, float minHeight, float maxHeight,
		std::vector< Eigen::Vector3f, Eigen::aligned_allocator< Eigen::Vector3f > > convexHull ) {

	const float min_resolution = minResolution;
	const float max_radius = 30.f;

	pcl::ExtractPolygonalPrismData< pcl::PointXYZRGBA > hull_limiter;

	// extract map and stitched point cloud from selected volume..
	// find convex hull for selected points in reference frame
	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud_selected_points( new pcl::PointCloud< pcl::PointXYZRGBA >() );
	for( unsigned int j = 0; j < convexHull.size(); j++ ) {
		pcl::PointXYZRGBA p;
		p.x = convexHull[ j ]( 0 );
		p.y = convexHull[ j ]( 1 );
		p.z = convexHull[ j ]( 2 );
		cloud_selected_points->points.push_back( p );
	}

	pcl::PointCloud< pcl::PointXYZRGBA >::Ptr cloud_convex_hull( new pcl::PointCloud< pcl::PointXYZRGBA >() );
	pcl::ConvexHull< pcl::PointXYZRGBA > chull;
	chull.setInputCloud( cloud_selected_points );
	chull.reconstruct( *cloud_convex_hull );

	boost::shared_ptr< MultiResolutionSurfelMap > graphmap = boost::shared_ptr< MultiResolutionSurfelMap >( new MultiResolutionSurfelMap( min_resolution, max_radius ) );
	graphmap->imageAllocator_ = imageAllocator_;

	for( unsigned int v_id = 0; v_id < optimizer_->vertices().size(); v_id++ ) {

		if( keyFrameNodeMap_.find( v_id ) == keyFrameNodeMap_.end() )
			continue;

		g2o::VertexSE3* v = dynamic_cast< g2o::VertexSE3* >( optimizer_->vertex( v_id ) );

		if( v->edges().size() == 0 && optimizer_->vertices().size() > 1 )
			continue;

		Eigen::Matrix4d v_pose = v->estimate().matrix();

		Eigen::Matrix4d transform = referenceTransform * v_pose;

		pcl::PointCloud< pcl::PointXYZRGBA >::Ptr transformedCloud = pcl::PointCloud< pcl::PointXYZRGBA >::Ptr( new pcl::PointCloud< pcl::PointXYZRGBA >() );
		pcl::transformPointCloud( *( keyFrameNodeMap_[ v_id ]->cloud_ ), *transformedCloud, transform.cast< float >() );

		transformedCloud->sensor_origin_ = transform.block< 4, 1 >( 0, 3 ).cast< float >();
		transformedCloud->sensor_orientation_ = Eigen::Quaternionf( transform.block< 3, 3 >( 0, 0 ).cast< float >() );

		// get indices in convex hull
		pcl::PointIndices::Ptr object_indices( new pcl::PointIndices() );
		hull_limiter.setInputCloud( transformedCloud );
		hull_limiter.setInputPlanarHull( cloud_convex_hull );
		hull_limiter.setHeightLimits( minHeight, maxHeight );
		hull_limiter.setViewPoint( transformedCloud->sensor_origin_[ 0 ], transformedCloud->sensor_origin_[ 1 ], transformedCloud->sensor_origin_[ 2 ] );
		hull_limiter.segment( *object_indices );

		std::cout << object_indices->indices.size() << "\n";

		// add keyframe to map
		graphmap->setApplyUpdate( false );
		graphmap->markUpdateImprovedEffViewDistSurfels( transformedCloud->sensor_origin_.block< 3, 1 >( 0, 0 ) );
		std::vector< int > imageBorderIndices;
		graphmap->findVirtualBorderPoints( *( keyFrameNodeMap_[ v_id ]->cloud_ ), imageBorderIndices );
		graphmap->markNoUpdateAtPoints( *transformedCloud, imageBorderIndices );
		graphmap->unevaluateSurfels();
		graphmap->addPoints( *transformedCloud, object_indices->indices );
		graphmap->clearUpdateSurfelsAtPoints( *transformedCloud, imageBorderIndices ); // only new surfels at these points have up_to_date == false !
		graphmap->octree_->root_->establishNeighbors();
		graphmap->clearUnstableSurfels();
		graphmap->setApplyUpdate( true );
		graphmap->evaluateSurfels();
		graphmap->buildShapeTextureFeatures();

	}

	return graphmap;

}

