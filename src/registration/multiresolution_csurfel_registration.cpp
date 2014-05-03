/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 16.05.2011
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

#include "mrsmap/registration/multiresolution_csurfel_registration.h"

#include <mrsmap/utilities/utilities.h>

#include <g2o/types/slam3d/dquat2mat.h>

#include <deque>

#include <fstream>

#include <tbb/tbb.h>

using namespace mrsmap;


#define ADD_SMOOTH_POS_COVARIANCE true
#define SMOOTH_SURFACE_COV_FACTOR 0.001f //  0.001f// just for numeric stability

#define SURFEL_MATCH_ANGLE_THRESHOLD  0.5 //0.5  // 0.5 ?
#define REGISTRATION_MIN_NUM_SURFELS 0
#define MAX_FEATURE_DIST2 0.1

#define MATCH_LIKELIHOOD_USE_COLOR 0
#define COLOR_DAMP_DIFF 0.05

#define PARALLEL 1


MultiResolutionColorSurfelRegistration::MultiResolutionColorSurfelRegistration() {

	use_prior_pose_ = false;
	prior_pose_mean_ = Eigen::Matrix< double, 6, 1 >::Zero();
	prior_pose_invcov_ = Eigen::Matrix< double, 6, 6 >::Identity();

}


void MultiResolutionColorSurfelRegistration::setPriorPose( bool enabled, const Eigen::Matrix< double, 6, 1 >& prior_pose_mean, const Eigen::Matrix< double, 6, 1 >& prior_pose_variances ) {

	use_prior_pose_ = enabled;
	prior_pose_mean_ = prior_pose_mean;
	prior_pose_invcov_ = Eigen::DiagonalMatrix< double, 6 >( prior_pose_variances ).inverse();

}



spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* MultiResolutionColorSurfelRegistration::calculateNegLogLikelihoodFeatureScoreN( double& logLikelihood, double& featureScore, bool& virtualBorder, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target, const Eigen::Matrix4d& transform, bool interpolate ) {

	// for each surfel in node with applyUpdate set and sufficient points, transform to target using transform,
	// then measure negative log likelihood

	const double normalStd = 0.5*0.125*M_PI;

	featureScore = std::numeric_limits<double>::max();
	logLikelihood = std::numeric_limits<double>::max();

	Eigen::Matrix3d rotation = transform.block<3,3>(0,0);

	// determine corresponding node in target..
	Eigen::Vector4f npos = node->getPosition();
	npos(3) = 1.0;
	Eigen::Vector4f npos_match_src = transform.cast<float>() * npos;

	if( !pointInImage( npos_match_src ) )
		virtualBorder = true;


	std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* > neighbors;
	neighbors.reserve(50);
	Eigen::Vector4f minPosition = npos_match_src - Eigen::Vector4f( 2.f*node->resolution(), 2.f*node->resolution(), 2.f*node->resolution(), 0.f );
	Eigen::Vector4f maxPosition = npos_match_src + Eigen::Vector4f( 2.f*node->resolution(), 2.f*node->resolution(), 2.f*node->resolution(), 0.f );

	target.octree_->getAllNodesInVolumeOnDepth( neighbors, minPosition, maxPosition, node->depth_, true );

	if( neighbors.size() == 0 ) {
		return NULL;
	}

	spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_matched = NULL;
	MultiResolutionColorSurfelMap::Surfel* srcSurfel = NULL;
	MultiResolutionColorSurfelMap::Surfel* matchedSurfel = NULL;
	int matchedSurfelIdx = -1;
	double bestDist = std::numeric_limits<double>::max();


	// get closest node in neighbor list
	for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

		MultiResolutionColorSurfelMap::Surfel& surfel = node->value_.surfels_[i];

		// border points are returned but must be handled later!
		if( surfel.num_points_ < MIN_SURFEL_POINTS ) {
			continue;
		}

		Eigen::Vector4d pos;
		pos.block<3,1>(0,0) = surfel.mean_.block<3,1>(0,0);
		pos(3,0) = 1.f;

		Eigen::Vector4d pos_match_src = transform * pos;
		Eigen::Vector3d dir_match_src = rotation * surfel.initial_view_dir_;

		for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >::iterator it = neighbors.begin(); it != neighbors.end(); it++ ) {

			if( (*it)->value_.border_ != node->value_.border_ )
				continue;

			MultiResolutionColorSurfelMap::Surfel* bestMatchSurfel = NULL;
			int bestMatchSurfelIdx = -1;
			double bestMatchDist = -1.f;
			for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {

				const MultiResolutionColorSurfelMap::Surfel& srcSurfel2 = (*it)->value_.surfels_[k];

				if( srcSurfel2.num_points_ < MIN_SURFEL_POINTS )
					continue;

				const double dist = dir_match_src.dot( srcSurfel2.initial_view_dir_ );
				if( dist >= SURFEL_MATCH_ANGLE_THRESHOLD && dist >= bestMatchDist ) {
					bestMatchSurfel = &((*it)->value_.surfels_[k]);
					bestMatchDist = dist;
					bestMatchSurfelIdx = k;
				}
			}

			if( bestMatchSurfel ) {
				// use distance between means
				double dist = (pos_match_src.block<3,1>(0,0) - bestMatchSurfel->mean_.block<3,1>(0,0)).norm();
				if( dist < bestDist ) {
					bestDist = dist;
					srcSurfel = &surfel;
					n_matched = *it;
					matchedSurfel = bestMatchSurfel;
					matchedSurfelIdx = bestMatchSurfelIdx;
				}
			}
		}

	}

	// border points are returned but must be handled later!
	if( !n_matched ) {
		return NULL;
	}

	if( !srcSurfel->applyUpdate_ || !matchedSurfel->applyUpdate_ )
		virtualBorder = true;

	featureScore = 0;//srcSurfel->agglomerated_shape_texture_features_.distance( matchedSurfel->agglomerated_shape_texture_features_ );

	Eigen::Vector4d pos;
	pos.block<3,1>(0,0) = srcSurfel->mean_.block<3,1>(0,0);
	pos(3,0) = 1.f;

	Eigen::Vector4d pos_match_src = transform * pos;

	double l = 0;


	if( MATCH_LIKELIHOOD_USE_COLOR ) {

		Eigen::Matrix< double, 6, 6 > rotation6 = Eigen::Matrix< double, 6, 6 >::Identity();
		rotation6.block<3,3>(0,0) = rotation;

		Eigen::Matrix< double, 6, 6 > cov1_ss;
		Eigen::Matrix< double, 6, 1 > dstMean;

		bool in_interpolation_range = true;

		if( interpolate ) {

			// use trilinear interpolation to handle discretization effects
			// => associate with neighbors and weight correspondences
			// only makes sense when match is within resolution distance to the node center
			const float resolution = node->resolution();

			// associate with neighbors for which distance to the node center is smaller than resolution

			dstMean.setZero();
			cov1_ss.setZero();

			double sumWeight = 0.f;
			double sumWeight2 = 0.f;

			for( int s = 0; s < 27; s++ ) {

				spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_dst_n = n_matched->neighbors_[s];

				if(!n_dst_n)
					continue;

				MultiResolutionColorSurfelMap::Surfel* dst_n = &n_dst_n->value_.surfels_[matchedSurfelIdx];
				if( dst_n->num_points_ < MIN_SURFEL_POINTS )
					continue;

				Eigen::Vector3d centerDiff_n = n_dst_n->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_match_src.block<3,1>(0,0);
				const double dx = resolution - fabsf(centerDiff_n(0));
				const double dy = resolution - fabsf(centerDiff_n(1));
				const double dz = resolution - fabsf(centerDiff_n(2));

				if( dx > 0 && dy > 0 && dz > 0 ) {

					const double weight = dx*dy*dz;

					dstMean += weight * dst_n->mean_;
					cov1_ss += weight*weight * (dst_n->cov_);

					sumWeight += weight;
					sumWeight2 += weight*weight;

				}


			}

			// numerically stable?
			if( sumWeight > resolution* 1e-6 ) {
				dstMean /= sumWeight;
				cov1_ss /= sumWeight2;

			}
			else
				in_interpolation_range = false;

		}

		if( !interpolate || !in_interpolation_range ) {

			dstMean = matchedSurfel->mean_;
			cov1_ss = matchedSurfel->cov_;

		}


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
		cov1_ss *= INTERPOLATION_COV_FACTOR;
		const Eigen::Matrix< double, 6, 6 > cov2_ss = INTERPOLATION_COV_FACTOR * srcSurfel->cov_;

		Eigen::Matrix< double, 6, 1 > diff_s;
		diff_s.block<3,1>(0,0) = dstMean.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0);
		diff_s.block<3,1>(3,0) = dstMean.block<3,1>(3,0) - srcSurfel->mean_.block<3,1>(3,0);
		if( fabs(diff_s(3)) < COLOR_DAMP_DIFF )
			diff_s(3) = 0;
		if( fabs(diff_s(4)) < COLOR_DAMP_DIFF )
			diff_s(4) = 0;
		if( fabs(diff_s(5)) < COLOR_DAMP_DIFF )
			diff_s(5) = 0;

		if( diff_s(3) < 0 )
			diff_s(3) += COLOR_DAMP_DIFF;
		if( diff_s(4) < 0 )
			diff_s(4) += COLOR_DAMP_DIFF;
		if( diff_s(5) < 0 )
			diff_s(5) += COLOR_DAMP_DIFF;

		if( diff_s(3) > 0 )
			diff_s(3) -= COLOR_DAMP_DIFF;
		if( diff_s(4) > 0 )
			diff_s(4) -= COLOR_DAMP_DIFF;
		if( diff_s(5) > 0 )
			diff_s(5) -= COLOR_DAMP_DIFF;

		const Eigen::Matrix< double, 6, 6 > Rcov2_ss = rotation6 * cov2_ss;

		const Eigen::Matrix< double, 6, 6 > cov_ss = cov1_ss + Rcov2_ss * rotation6.transpose();
		const Eigen::Matrix< double, 6, 6 > invcov_ss = cov_ss.inverse();

		const Eigen::Matrix< double, 6, 1 > invcov_ss_diff_s = invcov_ss * diff_s;

		l = log( cov_ss.determinant() ) + diff_s.dot(invcov_ss_diff_s);


	}
	else {

		Eigen::Matrix3d cov1_ss;
		Eigen::Vector3d dstMean;

		bool in_interpolation_range = true;

		if( interpolate ) {

			// use trilinear interpolation to handle discretization effects
			// => associate with neighbors and weight correspondences
			// only makes sense when match is within resolution distance to the node center
			const float resolution = node->resolution();

			// associate with neighbors for which distance to the node center is smaller than resolution

			dstMean.setZero();
			cov1_ss.setZero();

			double sumWeight = 0.f;
			double sumWeight2 = 0.f;

			for( int s = 0; s < 27; s++ ) {

				spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_dst_n = n_matched->neighbors_[s];

				if(!n_dst_n)
					continue;

				MultiResolutionColorSurfelMap::Surfel* dst_n = &n_dst_n->value_.surfels_[matchedSurfelIdx];
				if( dst_n->num_points_ < MIN_SURFEL_POINTS )
					continue;

				Eigen::Vector3d centerDiff_n = n_dst_n->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_match_src.block<3,1>(0,0);
				const double dx = resolution - fabsf(centerDiff_n(0));
				const double dy = resolution - fabsf(centerDiff_n(1));
				const double dz = resolution - fabsf(centerDiff_n(2));

				if( dx > 0 && dy > 0 && dz > 0 ) {

					const double weight = dx*dy*dz;

					dstMean += weight * dst_n->mean_.block<3,1>(0,0);
					cov1_ss += weight*weight * (dst_n->cov_.block<3,3>(0,0));

					sumWeight += weight;
					sumWeight2 += weight*weight;

				}


			}

			// numerically stable?
			if( sumWeight > resolution* 1e-6 ) {
				dstMean /= sumWeight;
				cov1_ss /= sumWeight2;

			}
			else
				in_interpolation_range = false;

		}

		if( !interpolate || !in_interpolation_range ) {

			dstMean = matchedSurfel->mean_.block<3,1>(0,0);
			cov1_ss = matchedSurfel->cov_.block<3,3>(0,0);

		}


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
		cov1_ss *= INTERPOLATION_COV_FACTOR;
		const Eigen::Matrix3d cov2_ss = INTERPOLATION_COV_FACTOR * srcSurfel->cov_.block<3,3>(0,0);

		const Eigen::Vector3d diff_s = dstMean - pos_match_src.block<3,1>(0,0);

		const Eigen::Matrix3d Rcov2_ss = rotation * cov2_ss;

		const Eigen::Matrix3d cov_ss = cov1_ss + Rcov2_ss * rotation.transpose();
		const Eigen::Matrix3d invcov_ss = cov_ss.inverse();

		const Eigen::Vector3d invcov_ss_diff_s = invcov_ss * diff_s;

		l = log( cov_ss.determinant() ) + diff_s.dot(invcov_ss_diff_s);

	}


	// also consider normal orientation in the likelihood

	Eigen::Vector4d normal_src;
	normal_src.block<3,1>(0,0) = srcSurfel->normal_;
	normal_src(3,0) = 0.0;
	normal_src = (transform * normal_src).eval();

	double normalError = std::min( 2.0 * normalStd, acos( normal_src.block<3,1>(0,0).dot( matchedSurfel->normal_ ) ) );
	double normalExponent = normalError * normalError / ( normalStd*normalStd );
	double normalLogLikelihood = log( 2.0 * M_PI * normalStd ) + normalExponent;

	l += normalLogLikelihood;

	logLikelihood = std::min( l, logLikelihood );

	return n_matched;

}


spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* MultiResolutionColorSurfelRegistration::calculateNegLogLikelihoodN( double& logLikelihood, bool& virtualBorder, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target, const Eigen::Matrix4d& transform, bool interpolate ) {

	double featureScore = 0.0;

	return calculateNegLogLikelihoodFeatureScoreN( logLikelihood, featureScore, virtualBorder, node, target, transform, interpolate );

}


bool MultiResolutionColorSurfelRegistration::calculateNegLogLikelihood( double& logLikelihood, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target, const Eigen::Matrix4d& transform, bool interpolate ) {

	bool virtualBorder = false;
	if( calculateNegLogLikelihoodN( logLikelihood, virtualBorder, node, target, transform, interpolate ) != NULL )
		return true;
	else
		return false;

}


void MultiResolutionColorSurfelRegistration::associateMapsBreadthFirstParallel( MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, algorithm::OcTreeSamplingVectorMap< float, MultiResolutionColorSurfelMap::NodeValue >& targetSamplingMap, Eigen::Matrix4d& transform, double minResolution, double maxResolution, double searchDistFactor, double maxSearchDist, bool useFeatures ) {


	target.distributeAssociatedFlag();

	int maxDepth = std::min( source.octree_->max_depth_, target.octree_->max_depth_ );

	// start at coarsest resolution
	// if all children associated, skip the node,
	// otherwise
	// - if already associated from previous iteration, search in local neighborhood
	// - if not associated in previous iteration, but parent has been associated, choose among children of parent's match
	// - otherwise, search in local volume for matches

	for( int d = maxDepth; d >= 0; d-- ) {

		const float processResolution = source.octree_->volumeSizeForDepth( d );

		if( processResolution < minResolution || processResolution > maxResolution ) {
			continue;
		}

		associateNodeListParallel( surfelAssociations, source, target, targetSamplingMap[d], d, transform, searchDistFactor, maxSearchDist, useFeatures );

	}


}


class AssociateFunctor {
public:
	AssociateFunctor( tbb::concurrent_vector< MultiResolutionColorSurfelRegistration::SurfelAssociation >* associations, MultiResolutionColorSurfelMap* source, MultiResolutionColorSurfelMap* target, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >* nodes, const Eigen::Matrix4d& transform, int processDepth, double searchDistFactor, double maxSearchDist, bool useFeatures ) {
		associations_ = associations;
		source_ = source;
		target_ = target;
		nodes_ = nodes;
		transform_ = transform;
		transformf_ = transform.cast<float>();
		rotation_ = transform.block<3,3>(0,0);

		process_depth_ = processDepth;
		process_resolution_ = source_->octree_->volumeSizeForDepth( processDepth );
		search_dist_ = std::min( searchDistFactor*process_resolution_, maxSearchDist );
		search_dist2_ = search_dist_*search_dist_;
		search_dist_vec_ = Eigen::Vector4f( search_dist_, search_dist_, search_dist_, 0.f );

		use_features_ = useFeatures;

		num_vol_queries_ = 0;
		num_finds_ = 0;
		num_neighbors_ = 0;


	}

	~AssociateFunctor() {}

	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*nodes_)[i]);
	}


	void operator()( spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >*& node ) const {

		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n = node;

		if( n->value_.associated_ == -1 )
			return;

		// all children associated?
		int numAssociatedChildren = 0;
		int numChildren = 0;
		for( unsigned int i = 0; i < 8; i++ ) {
			if( n->children_[i] ) {
				numChildren++;
				if( n->children_[i]->value_.associated_ == 1 )
					numAssociatedChildren++;
			}
		}

		if( numAssociatedChildren > 0 )
			n->value_.associated_ = 2;

		if( numChildren > 0 && numChildren == numAssociatedChildren )
			return;

		if( numChildren > 0 && numAssociatedChildren > 0 ) {
			return;
		}

		if( !n->value_.associated_ )
			return;


		// check if surfels exist and can be associated by view direction
		// use only one best association per node
		float bestAssocDist = std::numeric_limits<float>::max();
		float bestAssocFeatureDist = std::numeric_limits<float>::max();
		MultiResolutionColorSurfelRegistration::SurfelAssociation bestAssoc;

		bool hasSurfel = false;

		// TODO: collect features for view directions (surfels)
		// once a representative node is chosen, search for feature correspondences by sweeping up the tree up to a maximum search distance.
		// check compatibility using inverse depth parametrization

		// check if a surfels exist
		for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

			// if image border points fall into this node, we must check the children_
			if( !n->value_.surfels_[i].applyUpdate_ ) {
				continue;
			}

			if( n->value_.surfels_[i].num_points_ < MIN_SURFEL_POINTS ) {
				continue;
			}

			hasSurfel = true;
		}

		if( hasSurfel ) {

			spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src_last = NULL;
			std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* > neighbors;

			// association of this node exists from a previous iteration?
			char surfelSrcIdx = -1;
			char surfelDstIdx = -1;
			if( n->value_.association_ ) {
				n_src_last = n->value_.association_;
				surfelSrcIdx = n->value_.assocSurfelIdx_;
				surfelDstIdx = n->value_.assocSurfelDstIdx_;
				n_src_last->getNeighbors( neighbors );
			}

			// does association of parent exist from a previous iteration?
			if( !n_src_last ) {

				if( false && n->parent_ && n->parent_->value_.association_ ) {

					n_src_last = n->parent_->value_.association_;
					surfelSrcIdx = n->parent_->value_.assocSurfelIdx_;
					surfelDstIdx = n->parent_->value_.assocSurfelDstIdx_;

					Eigen::Vector4f npos = n->getCenterPosition();
					npos(3) = 1.f;
					Eigen::Vector4f npos_match_src = transformf_ * npos;

					n_src_last = n_src_last->findRepresentative( npos_match_src, process_depth_ );

					if( n_src_last )
						n_src_last->getNeighbors( neighbors );

				}
				else  {

					neighbors.reserve(50);

					Eigen::Vector4f npos = n->getCenterPosition();
					npos(3) = 1.f;
					Eigen::Vector4f npos_match_src = transformf_ * npos;

					// if direct look-up fails, perform a region query
					// in case there is nothing within the volume, the query will exit early

					Eigen::Vector4f minPosition = npos_match_src - search_dist_vec_;
					Eigen::Vector4f maxPosition = npos_match_src + search_dist_vec_;

					source_->octree_->getAllNodesInVolumeOnDepth( neighbors, minPosition, maxPosition, process_depth_, false );

				}

			}

			if( neighbors.size() == 0 ) {

				n->value_.association_ = NULL;
				n->value_.associated_ = 0;

				return;
			}


			if( surfelSrcIdx >= 0 && surfelDstIdx >= 0 ) {

				const MultiResolutionColorSurfelMap::Surfel& surfel = n->value_.surfels_[surfelSrcIdx];

				if( surfel.num_points_ >= MIN_SURFEL_POINTS ) {

					Eigen::Vector4d pos;
					pos.block<3,1>(0,0) = surfel.mean_.block<3,1>(0,0);
					pos(3,0) = 1.f;

					Eigen::Vector4d pos_match_src = transform_ * pos;
					Eigen::Vector3d dir_match_src = rotation_ * surfel.initial_view_dir_;

					// iterate through neighbors of the directly associated node to eventually find a better match
					for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >::iterator nit = neighbors.begin(); nit != neighbors.end(); ++nit ) {

						spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src = *nit;

						if( !n_src )
							continue;

						if( n->value_.border_ != n_src->value_.border_ )
							continue;


						// find matching surfel for the view direction, but allow to use a slightly worse fit,
						// when it is the only one with sufficient points for matching
						MultiResolutionColorSurfelMap::Surfel& dstSurfel = n_src->value_.surfels_[surfelDstIdx];

						if( dstSurfel.num_points_ < MIN_SURFEL_POINTS )
							continue;

						const double dist = dir_match_src.dot( dstSurfel.initial_view_dir_ );

						MultiResolutionColorSurfelMap::Surfel* bestMatchSurfel = NULL;
						int bestMatchSurfelIdx = -1;
						double bestMatchDist = -1.f;

						if( dist >= SURFEL_MATCH_ANGLE_THRESHOLD ) {
							bestMatchSurfel = &dstSurfel;
							bestMatchDist = dist;
							bestMatchSurfelIdx = surfelDstIdx;
						}

						if( !bestMatchSurfel ) {
							continue;
						}

						// calculate error metric for matching surfels
						double dist_pos2 = (bestMatchSurfel->mean_.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0)).squaredNorm();

						if( dist_pos2 > search_dist2_ )
							continue;

						// check local descriptor in any case
						float featureDist = 0.0;
						if( use_features_ )
							featureDist = surfel.agglomerated_shape_texture_features_.distance( bestMatchSurfel->agglomerated_shape_texture_features_ );
						if( use_features_ && featureDist > MAX_FEATURE_DIST2 )
							continue;

						float assocDist = sqrtf(dist_pos2);

						if( use_features_ )
							assocDist *= featureDist;

						if( assocDist < bestAssocDist ) {
							bestAssocDist = assocDist;
							bestAssocFeatureDist = featureDist;
							n->value_.surfels_[surfelSrcIdx].assocDist_ = assocDist;

							bestAssoc.n_src_ = n;
							bestAssoc.src_ = &n->value_.surfels_[surfelSrcIdx];
							bestAssoc.src_idx_ = surfelSrcIdx;
							bestAssoc.n_dst_ = n_src;
							bestAssoc.dst_ = bestMatchSurfel;
							bestAssoc.dst_idx_ = bestMatchSurfelIdx;
							bestAssoc.match = 1;

							if( use_features_ )
								bestAssoc.weight = MAX_FEATURE_DIST2 - featureDist;
							else
								bestAssoc.weight = 1.f;
						}

					}

				}

			}
			else {


				for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

					const MultiResolutionColorSurfelMap::Surfel& surfel = n->value_.surfels_[i];

					if( surfel.num_points_ < MIN_SURFEL_POINTS ) {
						continue;
					}

					// transform surfel mean with current transform and find corresponding node in source for current resolution
					// find corresponding surfel in node via the transformed view direction of the surfel

					Eigen::Vector4d pos;
					pos.block<3,1>(0,0) = surfel.mean_.block<3,1>(0,0);
					pos(3,0) = 1.f;

					Eigen::Vector4d pos_match_src = transform_ * pos;
					Eigen::Vector3d dir_match_src = rotation_ * surfel.initial_view_dir_;

					// iterate through neighbors of the directly associated node to eventually find a better match
					for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >::iterator nit = neighbors.begin(); nit != neighbors.end(); ++nit ) {

						spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src = *nit;

						if( !n_src )
							continue;

						if( n->value_.border_ != n_src->value_.border_ )
							continue;

						// find matching surfel for the view direction, but allow to use a slightly worse fit,
						// when it is the only one with sufficient points for matching
						MultiResolutionColorSurfelMap::Surfel* bestMatchSurfel = NULL;
						int bestMatchSurfelIdx = -1;
						double bestMatchDist = -1.f;
						for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {

							const MultiResolutionColorSurfelMap::Surfel& srcSurfel = n_src->value_.surfels_[k];

							if( srcSurfel.num_points_ < MIN_SURFEL_POINTS )
								continue;

							const double dist = dir_match_src.dot( srcSurfel.initial_view_dir_ );
							if( dist >= SURFEL_MATCH_ANGLE_THRESHOLD && dist >= bestMatchDist ) {
								bestMatchSurfel = &n_src->value_.surfels_[k];
								bestMatchDist = dist;
								bestMatchSurfelIdx = k;
							}
						}

						if( !bestMatchSurfel ) {
							continue;
						}

						// calculate error metric for matching surfels
						double dist_pos2 = (bestMatchSurfel->mean_.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0)).squaredNorm();

						if( dist_pos2 > search_dist2_ )
							continue;

						float featureDist = 0.f;
						if( use_features_) {
							featureDist = surfel.agglomerated_shape_texture_features_.distance( bestMatchSurfel->agglomerated_shape_texture_features_ );
							if( featureDist > MAX_FEATURE_DIST2 )
								continue;
						}


						float assocDist = sqrtf(dist_pos2);

						if( use_features_ )
							assocDist *= featureDist;

						if( assocDist < bestAssocDist ) {
							bestAssocDist = assocDist;
							bestAssocFeatureDist = featureDist;
							n->value_.surfels_[i].assocDist_ = assocDist;

							bestAssoc.n_src_ = n;
							bestAssoc.src_ = &n->value_.surfels_[i];
							bestAssoc.src_idx_ = i;
							bestAssoc.n_dst_ = n_src;
							bestAssoc.dst_ = bestMatchSurfel;
							bestAssoc.dst_idx_ = bestMatchSurfelIdx;
							bestAssoc.match = 1;

							if( use_features_ )
								bestAssoc.weight = MAX_FEATURE_DIST2 - featureDist;
							else
								bestAssoc.weight = 1.f;
						}

					}

				}

			}

		}

		if( bestAssocDist != std::numeric_limits<float>::max() ) {

			associations_->push_back( bestAssoc );
			n->value_.association_ = bestAssoc.n_dst_;
			n->value_.associated_ = 1;
			n->value_.assocSurfelIdx_ = bestAssoc.src_idx_;
			n->value_.assocSurfelDstIdx_ = bestAssoc.dst_idx_;

		}
		else {
			n->value_.association_ = NULL;
			n->value_.associated_ = 0;
		}


	}


	tbb::concurrent_vector< MultiResolutionColorSurfelRegistration::SurfelAssociation >* associations_;
	MultiResolutionColorSurfelMap* source_;
	MultiResolutionColorSurfelMap* target_;
	std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >* nodes_;
	Eigen::Matrix4d transform_;
	Eigen::Matrix4f transformf_;
	Eigen::Matrix3d rotation_;
	int process_depth_;
	float process_resolution_, search_dist_, search_dist2_;
	Eigen::Vector4f search_dist_vec_;
	bool use_features_;
	int num_vol_queries_, num_finds_, num_neighbors_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};


void MultiResolutionColorSurfelRegistration::associateNodeListParallel( MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >& nodes, int processDepth, Eigen::Matrix4d& transform, double searchDistFactor, double maxSearchDist, bool useFeatures ) {

	tbb::concurrent_vector< MultiResolutionColorSurfelRegistration::SurfelAssociation > depthAssociations;
	depthAssociations.reserve( nodes.size() );


	AssociateFunctor af( &depthAssociations, &source, &target, &nodes, transform, processDepth, searchDistFactor, maxSearchDist, useFeatures );

	if( PARALLEL )
		tbb::parallel_for_each( nodes.begin(), nodes.end(), af );
	else
		std::for_each( nodes.begin(), nodes.end(), af );


	surfelAssociations.insert( surfelAssociations.end(), depthAssociations.begin(), depthAssociations.end() );

}

class GradientFunctor {
public:
	GradientFunctor( MultiResolutionColorSurfelRegistration::SurfelAssociationList* assocList, double tx, double ty, double tz, double qx, double qy, double qz, double qw, bool relativeDerivatives, bool deriv2 = false, bool interpolate_neighbors = true, bool derivZ = false ) {

		assocList_ = assocList;

		const double inv_qw = 1.0 / qw;

		relativeDerivatives_ = relativeDerivatives;
		deriv2_ = deriv2;
		derivZ_ = derivZ;
		interpolate_neighbors_ = interpolate_neighbors;

		currentTransform.setIdentity();
		currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		currentTransform(0,3) = tx;
		currentTransform(1,3) = ty;
		currentTransform(2,3) = tz;

		currentRotation = Eigen::Matrix3d( currentTransform.block<3,3>(0,0) );
		currentRotationT = currentRotation.transpose();
		currentTranslation = Eigen::Vector3d( currentTransform.block<3,1>(0,3) );


		// build up derivatives of rotation and translation for the transformation variables
		dt_tx(0) = 1.f; dt_tx(1) = 0.f; dt_tx(2) = 0.f;
		dt_ty(0) = 0.f; dt_ty(1) = 1.f; dt_ty(2) = 0.f;
		dt_tz(0) = 0.f; dt_tz(1) = 0.f; dt_tz(2) = 1.f;


		if( relativeDerivatives_ ) {

			dR_qx.setZero();
			dR_qx(1,2) = -2;
			dR_qx(2,1) = 2;

			dR_qy.setZero();
			dR_qy(0,2) = 2;
			dR_qy(2,0) = -2;

			dR_qz.setZero();
			dR_qz(0,1) = -2;
			dR_qz(1,0) = 2;

		}
		else {

			// matrix(
			//  [ 0,
			//    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy),
			//    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qx,
			//    2*(qx^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz),
			//    2*(sqrt(-qz^2-qy^2-qx^2+1)-qx^2/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qx ]
			// )
			dR_qx(0,0) = 0.0;
			dR_qx(0,1) = 2.0*((qx*qz)*inv_qw+qy);
			dR_qx(0,2) = 2.0*(qz-(qx*qy)*inv_qw);
			dR_qx(1,0) = 2.0*(qy-(qx*qz)*inv_qw);
			dR_qx(1,1) = -4.0*qx;
			dR_qx(1,2) = 2.0*(qx*qx*inv_qw-qw);
			dR_qx(2,0) = 2.0*((qx*qy)*inv_qw+qz);
			dR_qx(2,1) = 2.0*(qw-qx*qx*inv_qw);
			dR_qx(2,2) = -4.0*qx;

			// matrix(
			//  [ -4*qy,
			//    2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
			//    2*(sqrt(-qz^2-qy^2-qx^2+1)-qy^2/sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    0,
			//    2*((qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qz) ],
			//  [ 2*(qy^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
			//    2*(qz-(qx*qy)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qy ]
			// )

			dR_qy(0,0) = -4.0*qy;
			dR_qy(0,1) = 2.0*((qy*qz)*inv_qw+qx);
			dR_qy(0,2) = 2.0*(qw-qy*qy*inv_qw);
			dR_qy(1,0) = 2.0*(qx-(qy*qz)*inv_qw);
			dR_qy(1,1) = 0.0;
			dR_qy(1,2) = 2.0*((qx*qy)*inv_qw+qz);
			dR_qy(2,0) = 2.0*(qy*qy*inv_qw-qw);
			dR_qy(2,1) = 2.0*(qz-(qx*qy)*inv_qw);
			dR_qy(2,2) = -4.0*qy;

			// matrix(
			//  [ -4*qz,
			//    2*(qz^2/sqrt(-qz^2-qy^2-qx^2+1)-sqrt(-qz^2-qy^2-qx^2+1)),
			//    2*(qx-(qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)) ],
			//  [ 2*(sqrt(-qz^2-qy^2-qx^2+1)-qz^2/sqrt(-qz^2-qy^2-qx^2+1)),
			//    -4*qz,
			//    2*((qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qy) ],
			//  [ 2*((qy*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qx),
			//    2*(qy-(qx*qz)/sqrt(-qz^2-qy^2-qx^2+1)),
			//    0 ]
			// )
			dR_qz(0,0) = -4.0*qz;
			dR_qz(0,1) = 2.0*(qz*qz*inv_qw-qw);
			dR_qz(0,2) = 2.0*(qx-(qy*qz)*inv_qw);
			dR_qz(1,0) = 2.0*(qw-qz*qz*inv_qw);
			dR_qz(1,1) = -4.0*qz;
			dR_qz(1,2) = 2.0*((qx*qz)*inv_qw+qy);
			dR_qz(2,0) = 2.0*((qy*qz)*inv_qw+qx);
			dR_qz(2,1) = 2.0*(qy-(qx*qz)*inv_qw);
			dR_qz(2,2) = 0.0;

		}


		dR_qxT = dR_qx.transpose();
		dR_qyT = dR_qy.transpose();
		dR_qzT = dR_qz.transpose();


		ddiff_s_tx.block<3,1>(0,0) = -dt_tx;
		ddiff_s_ty.block<3,1>(0,0) = -dt_ty;
		ddiff_s_tz.block<3,1>(0,0) = -dt_tz;

		if( deriv2_ ) {

			if( relativeDerivatives_ ) {

				d2R_qxx( 0, 0 ) = 0;
				d2R_qxx( 0, 1 ) = 0;
				d2R_qxx( 0, 2 ) = 0;
				d2R_qxx( 1, 0 ) = 0;
				d2R_qxx( 1, 1 ) = -4.0;
				d2R_qxx( 1, 2 ) = 0;
				d2R_qxx( 2, 0 ) = 0;
				d2R_qxx( 2, 1 ) = 0;
				d2R_qxx( 2, 2 ) = -4.0;

				d2R_qxy( 0, 0 ) = 0.0;
				d2R_qxy( 0, 1 ) = 2;
				d2R_qxy( 0, 2 ) = 0;
				d2R_qxy( 1, 0 ) = 2;
				d2R_qxy( 1, 1 ) = 0.0;
				d2R_qxy( 1, 2 ) = 0;
				d2R_qxy( 2, 0 ) = 0;
				d2R_qxy( 2, 1 ) = 0;
				d2R_qxy( 2, 2 ) = 0.0;

				d2R_qxz( 0, 0 ) = 0.0;
				d2R_qxz( 0, 1 ) = 0;
				d2R_qxz( 0, 2 ) = 2;
				d2R_qxz( 1, 0 ) = 0;
				d2R_qxz( 1, 1 ) = 0.0;
				d2R_qxz( 1, 2 ) = 0;
				d2R_qxz( 2, 0 ) = 2;
				d2R_qxz( 2, 1 ) = 0;
				d2R_qxz( 2, 2 ) = 0.0;

				d2R_qyy( 0, 0 ) = -4.0;
				d2R_qyy( 0, 1 ) = 0;
				d2R_qyy( 0, 2 ) = 0;
				d2R_qyy( 1, 0 ) = 0;
				d2R_qyy( 1, 1 ) = 0.0;
				d2R_qyy( 1, 2 ) = 0;
				d2R_qyy( 2, 0 ) = 0;
				d2R_qyy( 2, 1 ) = 0;
				d2R_qyy( 2, 2 ) = -4.0;

				d2R_qyz( 0, 0 ) = 0.0;
				d2R_qyz( 0, 1 ) = 0;
				d2R_qyz( 0, 2 ) = 0;
				d2R_qyz( 1, 0 ) = 0;
				d2R_qyz( 1, 1 ) = 0.0;
				d2R_qyz( 1, 2 ) = 2;
				d2R_qyz( 2, 0 ) = 0;
				d2R_qyz( 2, 1 ) = 2;
				d2R_qyz( 2, 2 ) = 0.0;

				d2R_qzz( 0, 0 ) = -4.0;
				d2R_qzz( 0, 1 ) = 0;
				d2R_qzz( 0, 2 ) = 0;
				d2R_qzz( 1, 0 ) = 0;
				d2R_qzz( 1, 1 ) = -4.0;
				d2R_qzz( 1, 2 ) = 0;
				d2R_qzz( 2, 0 ) = 0;
				d2R_qzz( 2, 1 ) = 0;
				d2R_qzz( 2, 2 ) = 0.0;

			}
			else {

				const double inv_qw3 = inv_qw*inv_qw*inv_qw;

				// matrix(
				// [ 0,
				//   2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4,
				//   2*((3*qx)/sqrt(-qz^2-qy^2-qx^2+1)+qx^3/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-(3*qx)/sqrt(-qz^2-qy^2-qx^2+1)-qx^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4 ] )
				d2R_qxx( 0, 0 ) = 0;
				d2R_qxx( 0, 1 ) = 2.0*(qz*inv_qw+qx*qx*qz*inv_qw3);
				d2R_qxx( 0, 2 ) = 2.0*(-qy*inv_qw-qx*qx*qy*inv_qw3);
				d2R_qxx( 1, 0 ) = 2.0*(-qz*inv_qw-qx*qx*qz*inv_qw3);
				d2R_qxx( 1, 1 ) = -4.0;
				d2R_qxx( 1, 2 ) = 2.0*(3.0*qx*inv_qw+qx*qx*qx*inv_qw3);
				d2R_qxx( 2, 0 ) = 2.0*(qy*inv_qw+qx*qx*qy*inv_qw3);
				d2R_qxx( 2, 1 ) = 2.0*(-3.0*qx*inv_qw-qx*qx*qx*inv_qw3);
				d2R_qxx( 2, 2 ) = -4.0;


				// matrix(
				// [ 0,
				//   2*((qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)+1),
				//   2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(1-(qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qy)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0 ] )
				d2R_qxy( 0, 0 ) = 0.0;
				d2R_qxy( 0, 1 ) = 2.0*(qx*qy*qz*inv_qw3+1.0);
				d2R_qxy( 0, 2 ) = 2.0*(-qx*inv_qw-qx*qy*qy*inv_qw3);
				d2R_qxy( 1, 0 ) = 2.0*(1.0-qx*qy*qz*inv_qw3);
				d2R_qxy( 1, 1 ) = 0.0;
				d2R_qxy( 1, 2 ) = 2.0*(qy*inv_qw+qx*qx*qy*inv_qw3);
				d2R_qxy( 2, 0 ) = 2.0*(qx*inv_qw+qx*qy*qy*inv_qw3);
				d2R_qxy( 2, 1 ) = 2.0*(-qy*inv_qw-qx*qx*qy*inv_qw3);
				d2R_qxy( 2, 2 ) = 0.0;


				// matrix(
				// [ 0,
				//   2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(1-(qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*((qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)+1),
				//   2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qx^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				// 0 ])
				d2R_qxz( 0, 0 ) = 0.0;
				d2R_qxz( 0, 1 ) = 2.0*(qx*inv_qw+qx*qz*qz*inv_qw3);
				d2R_qxz( 0, 2 ) = 2.0*(1.0-qx*qy*qz*inv_qw3);
				d2R_qxz( 1, 0 ) = 2.0*(-qx*inv_qw-qx*qz*qz*inv_qw3);
				d2R_qxz( 1, 1 ) = 0.0;
				d2R_qxz( 1, 2 ) = 2.0*(qz*inv_qw+qx*qx*qz*inv_qw3);
				d2R_qxz( 2, 0 ) = 2.0*(qx*qy*qz*inv_qw3+1.0);
				d2R_qxz( 2, 1 ) = 2.0*(-qz*inv_qw-qx*qx*qz*inv_qw3);
				d2R_qxz( 2, 2 ) = 0.0;

				// matrix(
				// [ -4,
				//   2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-(3*qy)/sqrt(-qz^2-qy^2-qx^2+1)-qy^3/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*((3*qy)/sqrt(-qz^2-qy^2-qx^2+1)+qy^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qy^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4 ])
				d2R_qyy( 0, 0 ) = -4.0;
				d2R_qyy( 0, 1 ) = 2.0*(qz*inv_qw+qy*qy*qz*inv_qw3);
				d2R_qyy( 0, 2 ) = 2.0*(-3.0*qy*inv_qw-qy*qy*qy*inv_qw3);
				d2R_qyy( 1, 0 ) = 2.0*(-qz*inv_qw-qy*qy*qz*inv_qw3);
				d2R_qyy( 1, 1 ) = 0.0;
				d2R_qyy( 1, 2 ) = 2.0*(qx*inv_qw+qx*qy*qy*inv_qw3);
				d2R_qyy( 2, 0 ) = 2.0*(3.0*qy*inv_qw+qy*qy*qy*inv_qw3);
				d2R_qyy( 2, 1 ) = 2.0*(-qx*inv_qw-qx*qy*qy*inv_qw3);
				d2R_qyy( 2, 2 ) = -4.0;

				// matrix(
				// [ 0,
				//   2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qz/sqrt(-qz^2-qy^2-qx^2+1)-(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0,
				//   2*((qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)+1) ],
				// [ 2*(qz/sqrt(-qz^2-qy^2-qx^2+1)+(qy^2*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(1-(qx*qy*qz)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0 ])
				d2R_qyz( 0, 0 ) = 0.0;
				d2R_qyz( 0, 1 ) = 2.0*(qy*inv_qw+qy*qz*qz*inv_qw3);
				d2R_qyz( 0, 2 ) = 2.0*(-qz*inv_qw-qy*qy*qz*inv_qw3);
				d2R_qyz( 1, 0 ) = 2.0*(-qy*inv_qw-qy*qz*qz*inv_qw3);
				d2R_qyz( 1, 1 ) = 0.0;
				d2R_qyz( 1, 2 ) = 2.0*(qx*qy*qz*inv_qw3+1.0);
				d2R_qyz( 2, 0 ) = 2.0*(qz*inv_qw+qy*qy*qz*inv_qw3);
				d2R_qyz( 2, 1 ) = 2.0*(1.0-qx*qy*qz*inv_qw3);
				d2R_qyz( 2, 2 ) = 0.0;

				// matrix(
				// [ -4,
				//   2*((3*qz)/sqrt(-qz^2-qy^2-qx^2+1)+qz^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qy/sqrt(-qz^2-qy^2-qx^2+1)-(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(-(3*qz)/sqrt(-qz^2-qy^2-qx^2+1)-qz^3/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   -4,
				//   2*(qx/sqrt(-qz^2-qy^2-qx^2+1)+(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)) ],
				// [ 2*(qy/sqrt(-qz^2-qy^2-qx^2+1)+(qy*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   2*(-qx/sqrt(-qz^2-qy^2-qx^2+1)-(qx*qz^2)/(-qz^2-qy^2-qx^2+1)^(3/2)),
				//   0 ])
				d2R_qzz( 0, 0 ) = -4.0;
				d2R_qzz( 0, 1 ) = 2.0*(3.0*qz*inv_qw+qz*qz*qz*inv_qw3);
				d2R_qzz( 0, 2 ) = 2.0*(-qy*inv_qw-qy*qz*qz*inv_qw3);
				d2R_qzz( 1, 0 ) = 2.0*(-3.0*qz*inv_qw-qz*qz*qz*inv_qw3);
				d2R_qzz( 1, 1 ) = -4.0;
				d2R_qzz( 1, 2 ) = 2.0*(qx*inv_qw+qx*qz*qz*inv_qw3);
				d2R_qzz( 2, 0 ) = 2.0*(qy*inv_qw+qy*qz*qz*inv_qw3);
				d2R_qzz( 2, 1 ) = 2.0*(-qx*inv_qw-qx*qz*qz*inv_qw3);
				d2R_qzz( 2, 2 ) = 0.0;

			}

			d2R_qxxT = d2R_qxx.transpose();
			d2R_qxyT = d2R_qxy.transpose();
			d2R_qxzT = d2R_qxz.transpose();
			d2R_qyyT = d2R_qyy.transpose();
			d2R_qyzT = d2R_qyz.transpose();
			d2R_qzzT = d2R_qzz.transpose();


			if( derivZ_ ) {

				// needed for the derivatives for the measurements

				ddiff_dzmx = Eigen::Vector3d( 1.0, 0.0, 0.0 );
				ddiff_dzmy = Eigen::Vector3d( 0.0, 1.0, 0.0 );
				ddiff_dzmz = Eigen::Vector3d( 0.0, 0.0, 1.0 );

				ddiff_dzsx = -currentRotation * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				ddiff_dzsy = -currentRotation * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				ddiff_dzsz = -currentRotation * Eigen::Vector3d( 0.0, 0.0, 1.0 );

				d2diff_qx_zsx = -dR_qx * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				d2diff_qx_zsy = -dR_qx * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				d2diff_qx_zsz = -dR_qx * Eigen::Vector3d( 0.0, 0.0, 1.0 );
				d2diff_qy_zsx = -dR_qy * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				d2diff_qy_zsy = -dR_qy * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				d2diff_qy_zsz = -dR_qy * Eigen::Vector3d( 0.0, 0.0, 1.0 );
				d2diff_qz_zsx = -dR_qz * Eigen::Vector3d( 1.0, 0.0, 0.0 );
				d2diff_qz_zsy = -dR_qz * Eigen::Vector3d( 0.0, 1.0, 0.0 );
				d2diff_qz_zsz = -dR_qz * Eigen::Vector3d( 0.0, 0.0, 1.0 );

			}

		}

	}

	~GradientFunctor() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*assocList_)[i]);
	}



	void operator()( MultiResolutionColorSurfelRegistration::SurfelAssociation& assoc ) const {


		if( assoc.match == 0 || !assoc.src_->applyUpdate_ || !assoc.dst_->applyUpdate_ ) {
			assoc.match = 0;
			return;
		}

		const float processResolution = assoc.n_src_->resolution();
		double weight = assoc.weight;

		Eigen::Vector4d pos;
		pos.block<3,1>(0,0) = assoc.src_->mean_.block<3,1>(0,0);
		pos(3,0) = 1.f;

		const Eigen::Vector4d pos_src = currentTransform * pos;

		double error = 0;

		double de_tx = 0;
		double de_ty = 0;
		double de_tz = 0;
		double de_qx = 0;
		double de_qy = 0;
		double de_qz = 0;

		Eigen::Matrix< double, 6, 6 > d2J_pp;
		Eigen::Matrix< double, 6, 6 > JSzJ;
		if( deriv2_ ) {
			d2J_pp.setZero();
			JSzJ.setZero();
		}



		// spatial component, marginalized

		Eigen::Matrix3d cov_ss_add;
		cov_ss_add.setZero();
		if( ADD_SMOOTH_POS_COVARIANCE ) {
			cov_ss_add.setIdentity();
			cov_ss_add *= SMOOTH_SURFACE_COV_FACTOR * processResolution*processResolution;
		}

		Eigen::Matrix3d cov1_ss;
		Eigen::Matrix3d cov2_ss = assoc.src_->cov_.block<3,3>(0,0) + cov_ss_add;

		Eigen::Vector3d dstMean;
		Eigen::Vector3d srcMean = assoc.src_->mean_.block<3,1>(0,0);

		bool in_interpolation_range = false;

		if( interpolate_neighbors_ ) {

			// use trilinear interpolation to handle discretization effects
			// => associate with neighbors and weight correspondences
			// only makes sense when match is within resolution distance to the node center
			const float resolution = processResolution;
			Eigen::Vector3d centerDiff = assoc.n_dst_->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_src.block<3,1>(0,0);
			if( resolution - fabsf(centerDiff(0)) > 0  && resolution - fabsf(centerDiff(1)) > 0  && resolution - fabsf(centerDiff(2)) > 0 ) {

				in_interpolation_range = true;

				// associate with neighbors for which distance to the node center is smaller than resolution

				dstMean.setZero();
				cov1_ss.setZero();

				double sumWeight = 0.f;
				double sumWeight2 = 0.f;

				for( int s = 0; s < 27; s++ ) {

					spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_dst_n = assoc.n_dst_->neighbors_[s];

					if(!n_dst_n)
						continue;

					MultiResolutionColorSurfelMap::Surfel* dst_n = &n_dst_n->value_.surfels_[assoc.dst_idx_];
					if( dst_n->num_points_ < MIN_SURFEL_POINTS )
						continue;

					Eigen::Vector3d centerDiff_n = n_dst_n->getCenterPosition().block<3,1>(0,0).cast<double>() - pos_src.block<3,1>(0,0);
					const double dx = resolution - fabsf(centerDiff_n(0));
					const double dy = resolution - fabsf(centerDiff_n(1));
					const double dz = resolution - fabsf(centerDiff_n(2));

					if( dx > 0 && dy > 0 && dz > 0 ) {

						const double weight = dx*dy*dz;

						dstMean += weight * dst_n->mean_.block<3,1>(0,0);
						cov1_ss += weight*weight * (dst_n->cov_.block<3,3>(0,0));

						sumWeight += weight;
						sumWeight2 += weight*weight;

					}


				}

				// numerically stable?
				if( sumWeight > resolution* 1e-6 ) {
					dstMean /= sumWeight;
					cov1_ss /= sumWeight2;

				}
				else
					in_interpolation_range = false;

				cov1_ss += cov_ss_add;


			}

		}

		if( !interpolate_neighbors_ || !in_interpolation_range ) {

			dstMean = assoc.dst_->mean_.block<3,1>(0,0);
			cov1_ss = assoc.dst_->cov_.block<3,3>(0,0) + cov_ss_add;

		}


		// has only marginal (positive!) effect on visual odometry result
		// makes tracking more robust (when only few surfels available)
		cov1_ss *= INTERPOLATION_COV_FACTOR;
		cov2_ss *= INTERPOLATION_COV_FACTOR;

		const Eigen::Vector3d TsrcMean = pos_src.block<3,1>(0,0);
		const Eigen::Vector3d diff_s = dstMean - TsrcMean;

		const Eigen::Matrix3d Rcov2_ss = currentRotation * cov2_ss;
		const Eigen::Matrix3d Rcov2_ssT = Rcov2_ss.transpose();

		const Eigen::Matrix3d cov_ss = cov1_ss + Rcov2_ss * currentRotationT;
		const Eigen::Matrix3d invcov_ss = cov_ss.inverse();
		const Eigen::Vector3d invcov_ss_diff_s = invcov_ss * diff_s;

		error = log( cov_ss.determinant() ) + diff_s.dot(invcov_ss_diff_s);


		if( relativeDerivatives_ ) {

			const Eigen::Matrix3d Rcov2R_ss = Rcov2_ss * currentRotationT;

			const Eigen::Vector3d ddiff_s_qx = -dR_qx * TsrcMean;
			const Eigen::Vector3d ddiff_s_qy = -dR_qy * TsrcMean;
			const Eigen::Vector3d ddiff_s_qz = -dR_qz * TsrcMean;



			const Eigen::Matrix3d dcov_ss_qx = dR_qx * Rcov2R_ss + Rcov2R_ss * dR_qx.transpose();
			const Eigen::Matrix3d dcov_ss_qy = dR_qy * Rcov2R_ss + Rcov2R_ss * dR_qy.transpose();
			const Eigen::Matrix3d dcov_ss_qz = dR_qz * Rcov2R_ss + Rcov2R_ss * dR_qz.transpose();

			const Eigen::Matrix3d dinvcov_ss_qx = -invcov_ss * dcov_ss_qx * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qy = -invcov_ss * dcov_ss_qy * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qz = -invcov_ss * dcov_ss_qz * invcov_ss;

			const Eigen::Vector3d dinvcov_ss_qx_diff_s = dinvcov_ss_qx * diff_s;
			const Eigen::Vector3d dinvcov_ss_qy_diff_s = dinvcov_ss_qy * diff_s;
			const Eigen::Vector3d dinvcov_ss_qz_diff_s = dinvcov_ss_qz * diff_s;


			de_tx = 2.0 * ddiff_s_tx.dot(invcov_ss_diff_s);
			de_ty = 2.0 * ddiff_s_ty.dot(invcov_ss_diff_s);
			de_tz = 2.0 * ddiff_s_tz.dot(invcov_ss_diff_s);
			de_qx = 2.0 * ddiff_s_qx.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qx_diff_s);
			de_qy = 2.0 * ddiff_s_qy.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qy_diff_s);
			de_qz = 2.0 * ddiff_s_qz.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qz_diff_s);

			// second term: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
			// -log( (2pi)^-(3/2) (det(cov))^(-1/2) )
			// = - log( (2pi)^-(3/2) ) - log( (det(cov))^(-1/2) )
			// = const. - (-0.5) * log( det(cov) )
			// = 0.5 * log( det(cov) ) => 0.5 factor can be left out also for the exp part...
			// d(log(det(cov)))/dq = 1/det(cov) * det(cov) * tr( cov^-1 * dcov/dq )
			// = tr( cov^-1 * dcov/dq )
			de_qx += (invcov_ss * dcov_ss_qx).trace();
			de_qy += (invcov_ss * dcov_ss_qy).trace();
			de_qz += (invcov_ss * dcov_ss_qz).trace();


			if( deriv2_ ) {

				const Eigen::Vector3d d2diff_s_qxx = -d2R_qxx * TsrcMean;
				const Eigen::Vector3d d2diff_s_qxy = -d2R_qxy * TsrcMean;
				const Eigen::Vector3d d2diff_s_qxz = -d2R_qxz * TsrcMean;
				const Eigen::Vector3d d2diff_s_qyy = -d2R_qyy * TsrcMean;
				const Eigen::Vector3d d2diff_s_qyz = -d2R_qyz * TsrcMean;
				const Eigen::Vector3d d2diff_s_qzz = -d2R_qzz * TsrcMean;

				const Eigen::Matrix3d d2cov_ss_qxx = d2R_qxx * Rcov2R_ss + 2.0 * dR_qx * Rcov2R_ss * dR_qxT + Rcov2R_ss * d2R_qxxT;
				const Eigen::Matrix3d d2cov_ss_qxy = d2R_qxy * Rcov2R_ss + dR_qx * Rcov2R_ss * dR_qyT + dR_qy * Rcov2R_ss * dR_qxT + Rcov2R_ss * d2R_qxyT;
				const Eigen::Matrix3d d2cov_ss_qxz = d2R_qxz * Rcov2R_ss + dR_qx * Rcov2R_ss * dR_qzT + dR_qz * Rcov2R_ss * dR_qxT + Rcov2R_ss * d2R_qxzT;
				const Eigen::Matrix3d d2cov_ss_qyy = d2R_qyy * Rcov2R_ss + 2.0 * dR_qy * Rcov2R_ss * dR_qyT + Rcov2R_ss * d2R_qyyT;
				const Eigen::Matrix3d d2cov_ss_qyz = d2R_qyz * Rcov2R_ss + dR_qy * Rcov2R_ss * dR_qzT + dR_qz * Rcov2R_ss * dR_qyT + Rcov2R_ss * d2R_qyzT;
				const Eigen::Matrix3d d2cov_ss_qzz = d2R_qzz * Rcov2R_ss + 2.0 * dR_qz * Rcov2R_ss * dR_qzT + Rcov2R_ss * d2R_qzzT;

				const Eigen::Matrix3d d2invcov_ss_qxx = -dinvcov_ss_qx * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxx * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qx;
				const Eigen::Matrix3d d2invcov_ss_qxy = -dinvcov_ss_qy * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxy * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qxz = -dinvcov_ss_qz * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxz * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qyy = -dinvcov_ss_qy * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyy * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qyz = -dinvcov_ss_qz * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyz * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qzz = -dinvcov_ss_qz * dcov_ss_qz * invcov_ss - invcov_ss * d2cov_ss_qzz * invcov_ss - invcov_ss * dcov_ss_qz * dinvcov_ss_qz;

				const Eigen::Vector3d invcov_ss_ddiff_s_tx = invcov_ss * ddiff_s_tx;
				const Eigen::Vector3d invcov_ss_ddiff_s_ty = invcov_ss * ddiff_s_ty;
				const Eigen::Vector3d invcov_ss_ddiff_s_tz = invcov_ss * ddiff_s_tz;
				const Eigen::Vector3d invcov_ss_ddiff_s_qx = invcov_ss * ddiff_s_qx;
				const Eigen::Vector3d invcov_ss_ddiff_s_qy = invcov_ss * ddiff_s_qy;
				const Eigen::Vector3d invcov_ss_ddiff_s_qz = invcov_ss * ddiff_s_qz;

				d2J_pp(0,0) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tx );
				d2J_pp(0,1) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(0,2) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(0,3) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(0,4) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(0,5) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(1,2) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(1,3) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(1,4) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(1,5) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) = 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(2,3) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(2,4) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(2,5) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(3,0) = d2J_pp(0,3);
				d2J_pp(3,1) = d2J_pp(1,3);
				d2J_pp(3,2) = d2J_pp(2,3);
				d2J_pp(3,3) = 2.0 * d2diff_s_qxx.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qx ) + diff_s.dot( d2invcov_ss_qxx * diff_s );
				d2J_pp(3,4) = 2.0 * d2diff_s_qxy.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxy * diff_s );
				d2J_pp(3,5) = 2.0 * d2diff_s_qxz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxz * diff_s );

				d2J_pp(4,0) = d2J_pp(0,4);
				d2J_pp(4,1) = d2J_pp(1,4);
				d2J_pp(4,2) = d2J_pp(2,4);
				d2J_pp(4,3) = d2J_pp(3,4);
				d2J_pp(4,4) = 2.0 * d2diff_s_qyy.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qy ) + diff_s.dot( d2invcov_ss_qyy * diff_s );
				d2J_pp(4,5) = 2.0 * d2diff_s_qyz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qy_diff_s ) + diff_s.dot( d2invcov_ss_qyz * diff_s );

				d2J_pp(5,0) = d2J_pp(0,5);
				d2J_pp(5,1) = d2J_pp(1,5);
				d2J_pp(5,2) = d2J_pp(2,5);
				d2J_pp(5,3) = d2J_pp(3,5);
				d2J_pp(5,4) = d2J_pp(4,5);
				d2J_pp(5,5) = 2.0 * d2diff_s_qzz.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qz.dot( invcov_ss_ddiff_s_qz ) + diff_s.dot( d2invcov_ss_qzz * diff_s );


				// further terms: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
				// = dtr( cov^-1 * dcov/dq ) / dq
				// = tr( d( cov^-1 * dcov/dq ) / dq )
				// = tr( dcov^-1/dq * dcov/dq + cov^-1 * d2cov/dqq )
				d2J_pp(0,0) += (dinvcov_ss_qx * dcov_ss_qx + invcov_ss * d2cov_ss_qxx).trace();
				d2J_pp(0,1) += (dinvcov_ss_qy * dcov_ss_qx + invcov_ss * d2cov_ss_qxy).trace();
				d2J_pp(0,2) += (dinvcov_ss_qz * dcov_ss_qx + invcov_ss * d2cov_ss_qxz).trace();
				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) += (dinvcov_ss_qy * dcov_ss_qy + invcov_ss * d2cov_ss_qyy).trace();
				d2J_pp(1,2) += (dinvcov_ss_qz * dcov_ss_qy + invcov_ss * d2cov_ss_qyz).trace();
				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) += (dinvcov_ss_qz * dcov_ss_qz + invcov_ss * d2cov_ss_qzz).trace();


				if( derivZ_ ) {

					// structure: pose along rows; first model coordinates, then scene
					Eigen::Matrix< double, 6, 3 > d2J_pzm, d2J_pzs;
					d2J_pzm(0,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(1,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(2,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(3,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qx_diff_s );
					d2J_pzs(3,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(3,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(3,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(4,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qy_diff_s );
					d2J_pzs(4,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(4,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(4,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(5,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qz_diff_s );
					d2J_pzs(5,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(5,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(5,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsz.dot( invcov_ss_diff_s );


					JSzJ += d2J_pzm * cov1_ss * d2J_pzm.transpose();
					JSzJ += d2J_pzs * cov2_ss * d2J_pzs.transpose();

				}

			}

		}
		else {

			const Eigen::Vector3d ddiff_s_qx = -dR_qx * srcMean;
			const Eigen::Vector3d ddiff_s_qy = -dR_qy * srcMean;
			const Eigen::Vector3d ddiff_s_qz = -dR_qz * srcMean;

			const Eigen::Matrix3d dcov_ss_qx = dR_qx * Rcov2_ssT + Rcov2_ss * dR_qxT;
			const Eigen::Matrix3d dcov_ss_qy = dR_qy * Rcov2_ssT + Rcov2_ss * dR_qyT;
			const Eigen::Matrix3d dcov_ss_qz = dR_qz * Rcov2_ssT + Rcov2_ss * dR_qzT;

			const Eigen::Matrix3d dinvcov_ss_qx = -invcov_ss * dcov_ss_qx * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qy = -invcov_ss * dcov_ss_qy * invcov_ss;
			const Eigen::Matrix3d dinvcov_ss_qz = -invcov_ss * dcov_ss_qz * invcov_ss;

			const Eigen::Vector3d dinvcov_ss_qx_diff_s = dinvcov_ss_qx * diff_s;
			const Eigen::Vector3d dinvcov_ss_qy_diff_s = dinvcov_ss_qy * diff_s;
			const Eigen::Vector3d dinvcov_ss_qz_diff_s = dinvcov_ss_qz * diff_s;


			de_tx = 2.0 * ddiff_s_tx.dot(invcov_ss_diff_s);
			de_ty = 2.0 * ddiff_s_ty.dot(invcov_ss_diff_s);
			de_tz = 2.0 * ddiff_s_tz.dot(invcov_ss_diff_s);
			de_qx = 2.0 * ddiff_s_qx.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qx_diff_s);
			de_qy = 2.0 * ddiff_s_qy.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qy_diff_s);
			de_qz = 2.0 * ddiff_s_qz.dot(invcov_ss_diff_s) + diff_s.dot(dinvcov_ss_qz_diff_s);

			// second term: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
			// -log( (2pi)^-(3/2) (det(cov))^(-1/2) )
			// = - log( (2pi)^-(3/2) ) - log( (det(cov))^(-1/2) )
			// = const. - (-0.5) * log( det(cov) )
			// = 0.5 * log( det(cov) ) => 0.5 factor can be left out also for the exp part...
			// d(log(det(cov)))/dq = 1/det(cov) * det(cov) * tr( cov^-1 * dcov/dq )
			// = tr( cov^-1 * dcov/dq )
			de_qx += (invcov_ss * dcov_ss_qx).trace();
			de_qy += (invcov_ss * dcov_ss_qy).trace();
			de_qz += (invcov_ss * dcov_ss_qz).trace();


			if( deriv2_ ) {

				const Eigen::Vector3d d2diff_s_qxx = -d2R_qxx * srcMean;
				const Eigen::Vector3d d2diff_s_qxy = -d2R_qxy * srcMean;
				const Eigen::Vector3d d2diff_s_qxz = -d2R_qxz * srcMean;
				const Eigen::Vector3d d2diff_s_qyy = -d2R_qyy * srcMean;
				const Eigen::Vector3d d2diff_s_qyz = -d2R_qyz * srcMean;
				const Eigen::Vector3d d2diff_s_qzz = -d2R_qzz * srcMean;

				const Eigen::Matrix3d d2cov_ss_qxx = d2R_qxx * Rcov2_ssT + 2.0 * dR_qx * cov2_ss * dR_qxT + Rcov2_ss * d2R_qxxT;
				const Eigen::Matrix3d d2cov_ss_qxy = d2R_qxy * Rcov2_ssT + dR_qx * cov2_ss * dR_qyT + dR_qy * cov2_ss * dR_qxT + Rcov2_ss * d2R_qxyT;
				const Eigen::Matrix3d d2cov_ss_qxz = d2R_qxz * Rcov2_ssT + dR_qx * cov2_ss * dR_qzT + dR_qz * cov2_ss * dR_qxT + Rcov2_ss * d2R_qxzT;
				const Eigen::Matrix3d d2cov_ss_qyy = d2R_qyy * Rcov2_ssT + 2.0 * dR_qy * cov2_ss * dR_qyT + Rcov2_ss * d2R_qyyT;
				const Eigen::Matrix3d d2cov_ss_qyz = d2R_qyz * Rcov2_ssT + dR_qy * cov2_ss * dR_qzT + dR_qz * cov2_ss * dR_qyT + Rcov2_ss * d2R_qyzT;
				const Eigen::Matrix3d d2cov_ss_qzz = d2R_qzz * Rcov2_ssT + 2.0 * dR_qz * cov2_ss * dR_qzT + Rcov2_ss * d2R_qzzT;

				const Eigen::Matrix3d d2invcov_ss_qxx = -dinvcov_ss_qx * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxx * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qx;
				const Eigen::Matrix3d d2invcov_ss_qxy = -dinvcov_ss_qy * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxy * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qxz = -dinvcov_ss_qz * dcov_ss_qx * invcov_ss - invcov_ss * d2cov_ss_qxz * invcov_ss - invcov_ss * dcov_ss_qx * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qyy = -dinvcov_ss_qy * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyy * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qy;
				const Eigen::Matrix3d d2invcov_ss_qyz = -dinvcov_ss_qz * dcov_ss_qy * invcov_ss - invcov_ss * d2cov_ss_qyz * invcov_ss - invcov_ss * dcov_ss_qy * dinvcov_ss_qz;
				const Eigen::Matrix3d d2invcov_ss_qzz = -dinvcov_ss_qz * dcov_ss_qz * invcov_ss - invcov_ss * d2cov_ss_qzz * invcov_ss - invcov_ss * dcov_ss_qz * dinvcov_ss_qz;

				const Eigen::Vector3d invcov_ss_ddiff_s_tx = invcov_ss * ddiff_s_tx;
				const Eigen::Vector3d invcov_ss_ddiff_s_ty = invcov_ss * ddiff_s_ty;
				const Eigen::Vector3d invcov_ss_ddiff_s_tz = invcov_ss * ddiff_s_tz;
				const Eigen::Vector3d invcov_ss_ddiff_s_qx = invcov_ss * ddiff_s_qx;
				const Eigen::Vector3d invcov_ss_ddiff_s_qy = invcov_ss * ddiff_s_qy;
				const Eigen::Vector3d invcov_ss_ddiff_s_qz = invcov_ss * ddiff_s_qz;

				d2J_pp(0,0) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tx );
				d2J_pp(0,1) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(0,2) = 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(0,3) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(0,4) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(0,5) = 2.0 * ddiff_s_tx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tx.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_ty );
				d2J_pp(1,2) = 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(1,3) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(1,4) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(1,5) = 2.0 * ddiff_s_ty.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_ty.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) = 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_tz );
				d2J_pp(2,3) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qx );
				d2J_pp(2,4) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qy );
				d2J_pp(2,5) = 2.0 * ddiff_s_tz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_tz.dot( invcov_ss_ddiff_s_qz );

				d2J_pp(3,0) = d2J_pp(0,3);
				d2J_pp(3,1) = d2J_pp(1,3);
				d2J_pp(3,2) = d2J_pp(2,3);
				d2J_pp(3,3) = 2.0 * d2diff_s_qxx.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qx ) + diff_s.dot( d2invcov_ss_qxx * diff_s );
				d2J_pp(3,4) = 2.0 * d2diff_s_qxy.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxy * diff_s );
				d2J_pp(3,5) = 2.0 * d2diff_s_qxz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qx_diff_s ) + diff_s.dot( d2invcov_ss_qxz * diff_s );

				d2J_pp(4,0) = d2J_pp(0,4);
				d2J_pp(4,1) = d2J_pp(1,4);
				d2J_pp(4,2) = d2J_pp(2,4);
				d2J_pp(4,3) = d2J_pp(3,4);
				d2J_pp(4,4) = 2.0 * d2diff_s_qyy.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qy ) + diff_s.dot( d2invcov_ss_qyy * diff_s );
				d2J_pp(4,5) = 2.0 * d2diff_s_qyz.dot( invcov_ss_diff_s ) + 2.0 * ddiff_s_qy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_s_qz.dot( dinvcov_ss_qy_diff_s ) + diff_s.dot( d2invcov_ss_qyz * diff_s );

				d2J_pp(5,0) = d2J_pp(0,5);
				d2J_pp(5,1) = d2J_pp(1,5);
				d2J_pp(5,2) = d2J_pp(2,5);
				d2J_pp(5,3) = d2J_pp(3,5);
				d2J_pp(5,4) = d2J_pp(4,5);
				d2J_pp(5,5) = 2.0 * d2diff_s_qzz.dot( invcov_ss_diff_s ) + 4.0 * ddiff_s_qz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * ddiff_s_qz.dot( invcov_ss_ddiff_s_qz ) + diff_s.dot( d2invcov_ss_qzz * diff_s );


				// further terms: derivative for normalizer of the normal distribution! det(cov) is not independent of q!
				// = dtr( cov^-1 * dcov/dq ) / dq
				// = tr( d( cov^-1 * dcov/dq ) / dq )
				// = tr( dcov^-1/dq * dcov/dq + cov^-1 * d2cov/dqq )
				d2J_pp(0,0) += (dinvcov_ss_qx * dcov_ss_qx + invcov_ss * d2cov_ss_qxx).trace();
				d2J_pp(0,1) += (dinvcov_ss_qy * dcov_ss_qx + invcov_ss * d2cov_ss_qxy).trace();
				d2J_pp(0,2) += (dinvcov_ss_qz * dcov_ss_qx + invcov_ss * d2cov_ss_qxz).trace();
				d2J_pp(1,0) = d2J_pp(0,1);
				d2J_pp(1,1) += (dinvcov_ss_qy * dcov_ss_qy + invcov_ss * d2cov_ss_qyy).trace();
				d2J_pp(1,2) += (dinvcov_ss_qz * dcov_ss_qy + invcov_ss * d2cov_ss_qyz).trace();
				d2J_pp(2,0) = d2J_pp(0,2);
				d2J_pp(2,1) = d2J_pp(1,2);
				d2J_pp(2,2) += (dinvcov_ss_qz * dcov_ss_qz + invcov_ss * d2cov_ss_qzz).trace();


				if( derivZ_ ) {

					// structure: pose along rows; first model coordinates, then scene
					Eigen::Matrix< double, 6, 3 > d2J_pzm, d2J_pzs;
					d2J_pzm(0,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(0,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tx );
					d2J_pzs(0,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tx );
					d2J_pzm(1,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(1,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_ty );
					d2J_pzs(1,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_ty );
					d2J_pzm(2,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(2,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_tz );
					d2J_pzs(2,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_tz );
					d2J_pzm(3,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qx_diff_s );
					d2J_pzm(3,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qx_diff_s );
					d2J_pzs(3,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(3,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(3,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qx ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qx_diff_s ) + 2.0 * d2diff_qx_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(4,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qy_diff_s );
					d2J_pzm(4,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qy_diff_s );
					d2J_pzs(4,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(4,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(4,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qy ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qy_diff_s ) + 2.0 * d2diff_qy_zsz.dot( invcov_ss_diff_s );
					d2J_pzm(5,0) = 2.0 * ddiff_dzmx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmx.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,1) = 2.0 * ddiff_dzmy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmy.dot( dinvcov_ss_qz_diff_s );
					d2J_pzm(5,2) = 2.0 * ddiff_dzmz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzmz.dot( dinvcov_ss_qz_diff_s );
					d2J_pzs(5,0) = 2.0 * ddiff_dzsx.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsx.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsx.dot( invcov_ss_diff_s );
					d2J_pzs(5,1) = 2.0 * ddiff_dzsy.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsy.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsy.dot( invcov_ss_diff_s );
					d2J_pzs(5,2) = 2.0 * ddiff_dzsz.dot( invcov_ss_ddiff_s_qz ) + 2.0 * ddiff_dzsz.dot( dinvcov_ss_qz_diff_s ) + 2.0 * d2diff_qz_zsz.dot( invcov_ss_diff_s );


					JSzJ += d2J_pzm * cov1_ss * d2J_pzm.transpose();
					JSzJ += d2J_pzs * cov2_ss * d2J_pzs.transpose();

				}

			}

		}


		assoc.df_tx = de_tx;
		assoc.df_ty = de_ty;
		assoc.df_tz = de_tz;
		assoc.df_qx = de_qx;
		assoc.df_qy = de_qy;
		assoc.df_qz = de_qz;

		if( deriv2_ ) {
			assoc.d2f = d2J_pp;

			if( derivZ_ )
				assoc.JSzJ = JSzJ;
		}

		assoc.error = error;
		assoc.weight = weight;
		assoc.match = 1;

		assert( !isnan(error) );




	}


	double tx, ty, tz, qx, qy, qz, qw;
	Eigen::Matrix4d currentTransform;
	Eigen::Vector3d ddiff_s_tx, ddiff_s_ty, ddiff_s_tz;
	Eigen::Matrix3d dR_qx, dR_qy, dR_qz;
	Eigen::Matrix3d dR_qxT, dR_qyT, dR_qzT;
	Eigen::Vector3d dt_tx, dt_ty, dt_tz;
	Eigen::Matrix3d currentRotation;
	Eigen::Matrix3d currentRotationT;
	Eigen::Vector3d currentTranslation;

	// 2nd order derivatives
	Eigen::Matrix3d d2R_qxx, d2R_qxy, d2R_qxz, d2R_qyy, d2R_qyz, d2R_qzz;
	Eigen::Matrix3d d2R_qxxT, d2R_qxyT, d2R_qxzT, d2R_qyyT, d2R_qyzT, d2R_qzzT;

	// 1st and 2nd order derivatives on Z
	Eigen::Vector3d ddiff_dzsx, ddiff_dzsy, ddiff_dzsz;
	Eigen::Vector3d ddiff_dzmx, ddiff_dzmy, ddiff_dzmz;
	Eigen::Vector3d d2diff_qx_zsx, d2diff_qx_zsy, d2diff_qx_zsz;
	Eigen::Vector3d d2diff_qy_zsx, d2diff_qy_zsy, d2diff_qy_zsz;
	Eigen::Vector3d d2diff_qz_zsx, d2diff_qz_zsy, d2diff_qz_zsz;


	bool relativeDerivatives_;
	bool deriv2_, derivZ_;
	bool interpolate_neighbors_;

	MultiResolutionColorSurfelRegistration::SurfelAssociationList* assocList_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};


class GradientFunctorLM {
public:


	GradientFunctorLM( MultiResolutionColorSurfelRegistration::SurfelAssociationList* assocList, double tx, double ty, double tz, double qx, double qy, double qz, double qw, bool derivs ) {

		derivs_ = derivs;

		assocList_ = assocList;

		currentTransform.setIdentity();
		currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		currentTransform(0,3) = tx;
		currentTransform(1,3) = ty;
		currentTransform(2,3) = tz;

		currentRotation = Eigen::Matrix3d( currentTransform.block<3,3>(0,0) );
		currentRotationT = currentRotation.transpose();
		currentTranslation = Eigen::Vector3d( currentTransform.block<3,1>(0,3) );

		if( derivs ) {

			const double inv_qw = 1.0 / qw;

			// build up derivatives of rotation and translation for the transformation variables
			dt_tx(0) = 1.f; dt_tx(1) = 0.f; dt_tx(2) = 0.f;
			dt_ty(0) = 0.f; dt_ty(1) = 1.f; dt_ty(2) = 0.f;
			dt_tz(0) = 0.f; dt_tz(1) = 0.f; dt_tz(2) = 1.f;

			dR_qx.setZero();
			dR_qx(1,2) = -2;
			dR_qx(2,1) = 2;

			dR_qy.setZero();
			dR_qy(0,2) = 2;
			dR_qy(2,0) = -2;

			dR_qz.setZero();
			dR_qz(0,1) = -2;
			dR_qz(1,0) = 2;

		}

	}

	~GradientFunctorLM() {}


	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for( size_t i=r.begin(); i!=r.end(); ++i )
			(*this)((*assocList_)[i]);
	}



	void operator()( MultiResolutionColorSurfelRegistration::SurfelAssociation& assoc ) const {


		if( assoc.match == 0 || !assoc.src_->applyUpdate_ || !assoc.dst_->applyUpdate_ ) {
			assoc.match = 0;
			return;
		}

		const Eigen::Matrix3d cov1_ss = assoc.dst_->cov_.block<3,3>(0,0);// + cov_ss_add;
		const Eigen::Matrix3d cov2_ss = assoc.src_->cov_.block<3,3>(0,0);// + cov_ss_add;

		const Eigen::Vector3d dstMean = assoc.dst_->mean_.block<3,1>(0,0);
		const Eigen::Vector3d srcMean = assoc.src_->mean_.block<3,1>(0,0);

		Eigen::Vector4d pos;
		pos.block<3,1>(0,0) = srcMean;
		pos(3,0) = 1.f;

		const Eigen::Vector4d pos_src = currentTransform * pos;

		const Eigen::Vector3d p_s = pos_src.block<3,1>(0,0);
		const Eigen::Vector3d diff_s = dstMean - p_s;

		const Eigen::Matrix3d cov_ss = INTERPOLATION_COV_FACTOR * (cov1_ss + currentRotation * cov2_ss * currentRotationT);
		const Eigen::Matrix3d invcov_ss = cov_ss.inverse();

		assoc.error = diff_s.dot(invcov_ss * diff_s);

		assoc.z = dstMean;
		assoc.f = p_s;

		if( derivs_ ) {

			assoc.df_dx.block<3,1>(0,0) = dt_tx;
			assoc.df_dx.block<3,1>(0,1) = dt_ty;
			assoc.df_dx.block<3,1>(0,2) = dt_tz;
			assoc.df_dx.block<3,1>(0,3) = dR_qx * pos_src.block<3,1>(0,0);
			assoc.df_dx.block<3,1>(0,4) = dR_qy * pos_src.block<3,1>(0,0);
			assoc.df_dx.block<3,1>(0,5) = dR_qz * pos_src.block<3,1>(0,0);

			assoc.W = invcov_ss;

		}


		assoc.match = 1;

	}



	Eigen::Matrix4d currentTransform;

	Eigen::Vector3d currentTranslation;
	Eigen::Vector3d dt_tx, dt_ty, dt_tz;

	Eigen::Matrix3d currentRotation, currentRotationT;
	Eigen::Matrix3d dR_qx, dR_qy, dR_qz;


	MultiResolutionColorSurfelRegistration::SurfelAssociationList* assocList_;

	bool derivs_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};



bool MultiResolutionColorSurfelRegistration::registrationErrorFunctionWithFirstDerivative( const Eigen::Matrix< double, 6, 1 >& x, void* params, double* f, Eigen::Matrix< double, 6, 1 >& df, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations ) {

	double sumError = 0.0;
	double sumWeight = 0.0;

	df.setZero();

	MultiResolutionColorSurfelRegistration::RegistrationFunctionParameters* p = (MultiResolutionColorSurfelRegistration::RegistrationFunctionParameters*) params;

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = p->lastWSign * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	GradientFunctor gf( &surfelAssociations, tx, ty, tz, qx, qy, qz, qw, false, false, p->interpolate_neighbors );


	if( PARALLEL )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );


	double numMatches = 0;
	for( MultiResolutionColorSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;

		float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
		float weight = nweight * it->weight;

		df(0) += weight * it->df_tx;
		df(1) += weight * it->df_ty;
		df(2) += weight * it->df_tz;
		df(3) += weight * it->df_qx;
		df(4) += weight * it->df_qy;
		df(5) += weight * it->df_qz;
		sumError += weight * it->error;
		sumWeight += weight;
		numMatches += 1.0;//nweight;

	}

	if( sumWeight <= 1e-10 ) {
		sumError = std::numeric_limits<double>::max();
		return false;
	}
	else if( numMatches < REGISTRATION_MIN_NUM_SURFELS ) {
		sumError = std::numeric_limits<double>::max();
		return false;
	}
	else {
		sumError /= sumWeight;
		df /= sumWeight;
	}

	if( use_prior_pose_ ) {

		df += 2.0 * prior_pose_invcov_ * (x - prior_pose_mean_);
	}

	*f = sumError;
	return true;

}



bool MultiResolutionColorSurfelRegistration::registrationErrorFunctionWithFirstAndSecondDerivative( const Eigen::Matrix< double, 6, 1 >& x, bool relativeDerivative, void* params, double* f, Eigen::Matrix< double, 6, 1 >& df, Eigen::Matrix< double, 6, 6 >& d2f, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations ) {

	double sumError = 0.0;
	double sumWeight = 0.0;

	df.setZero();
	d2f.setZero();

	MultiResolutionColorSurfelRegistration::RegistrationFunctionParameters* p = (MultiResolutionColorSurfelRegistration::RegistrationFunctionParameters*) params;

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = p->lastWSign * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	GradientFunctor gf( &surfelAssociations, tx, ty, tz, qx, qy, qz, qw, relativeDerivative, true, p->interpolate_neighbors );

	if( PARALLEL )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	int cidx = 0;
	if( p->correspondences_source_points_ ) {
		p->correspondences_source_points_->points.resize(surfelAssociations.size());
		p->correspondences_target_points_->points.resize(surfelAssociations.size());
	}


	double numMatches = 0;
	for( MultiResolutionColorSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;


		Eigen::Matrix< double, 6, 1 > dfloc;
		dfloc(0) = it->df_tx;
		dfloc(1) = it->df_ty;
		dfloc(2) = it->df_tz;
		dfloc(3) = it->df_qx;
		dfloc(4) = it->df_qy;
		dfloc(5) = it->df_qz;


		float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
		float weight = nweight * it->weight;

		df(0) += weight * it->df_tx;
		df(1) += weight * it->df_ty;
		df(2) += weight * it->df_tz;
		df(3) += weight * it->df_qx;
		df(4) += weight * it->df_qy;
		df(5) += weight * it->df_qz;
		d2f += weight * it->d2f;
		sumError += weight * it->error;
		sumWeight += weight;
		numMatches += 1.0;//nweight;



		if( p->correspondences_source_points_ ) {

			pcl::PointXYZRGBA& p1 = p->correspondences_source_points_->points[cidx];
			pcl::PointXYZRGBA& p2 = p->correspondences_target_points_->points[cidx];

			Eigen::Vector4f pos1 = it->n_dst_->getCenterPosition();
			Eigen::Vector4f pos2 = it->n_src_->getCenterPosition();

			p1.x = pos1(0);
			p1.y = pos1(1);
			p1.z = pos1(2);

			p1.r = nweight * 255.f;
			p1.g = 0;
			p1.b = (1.f-nweight) * 255.f;

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = pos2.block<3,1>(0,0).cast<double>();
			pos(3,0) = 1.f;

			const Eigen::Vector4d pos_src = gf.currentTransform * pos;

			p2.x = pos_src[0];
			p2.y = pos_src[1];
			p2.z = pos_src[2];

			p2.r = nweight * 255.f;
			p2.g = 0;
			p2.b = (1.f-nweight) * 255.f;

			cidx++;
		}

	}


	if( p->correspondences_source_points_ ) {
		p->correspondences_source_points_->points.resize(cidx);
		p->correspondences_target_points_->points.resize(cidx);
	}

	if( sumWeight <= 1e-10 ) {
		sumError = std::numeric_limits<double>::max();
		return false;
	}
	else if( numMatches < REGISTRATION_MIN_NUM_SURFELS ) {
		sumError = std::numeric_limits<double>::max();
		std::cout << "not enough surfels for robust matching " << numMatches << "\n";
		return false;
	}
	else {
		sumError /= sumWeight;
		df /= sumWeight;
		d2f /= sumWeight;
	}


	if( use_prior_pose_ ) {

		df += 2.0 * prior_pose_invcov_ * (x - prior_pose_mean_);
		d2f += 2.0 * prior_pose_invcov_;
	}



	*f = sumError;
	return true;

}



bool MultiResolutionColorSurfelRegistration::registrationErrorFunctionLM( const Eigen::Matrix< double, 6, 1 >& x, MultiResolutionColorSurfelRegistration::RegistrationFunctionParameters& params, double& f, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations ) {

	double sumError = 0.0;
	double sumWeight = 0.0;

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = params.lastWSign * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	GradientFunctorLM gf( &surfelAssociations, tx, ty, tz, qx, qy, qz, qw, false );

	if( PARALLEL )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	int cidx = 0;
	if( params.correspondences_source_points_ ) {
		params.correspondences_source_points_->points.resize(surfelAssociations.size());
		params.correspondences_target_points_->points.resize(surfelAssociations.size());
	}


	double numMatches = 0;
	for( MultiResolutionColorSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;


		float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
		float weight = nweight * it->weight;

		sumError += weight * it->error;
		sumWeight += weight;
		numMatches += 1.0;//nweight;



		if( params.correspondences_source_points_ ) {

			pcl::PointXYZRGBA& p1 = params.correspondences_source_points_->points[cidx];
			pcl::PointXYZRGBA& p2 = params.correspondences_target_points_->points[cidx];

			Eigen::Vector4f pos1 = it->n_dst_->getCenterPosition();
			Eigen::Vector4f pos2 = it->n_src_->getCenterPosition();

			p1.x = pos1(0);
			p1.y = pos1(1);
			p1.z = pos1(2);

			p1.r = nweight * 255.f;
			p1.g = 0;
			p1.b = (1.f-nweight) * 255.f;

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = pos2.block<3,1>(0,0).cast<double>();
			pos(3,0) = 1.f;

			const Eigen::Vector4d pos_src = gf.currentTransform * pos;

			p2.x = pos_src[0];
			p2.y = pos_src[1];
			p2.z = pos_src[2];

			p2.r = nweight * 255.f;
			p2.g = 0;
			p2.b = (1.f-nweight) * 255.f;

			cidx++;
		}

	}


	if( params.correspondences_source_points_ ) {
		params.correspondences_source_points_->points.resize(cidx);
		params.correspondences_target_points_->points.resize(cidx);
	}

	if( sumWeight <= 1e-10 ) {
		sumError = std::numeric_limits<double>::max();
		return false;
	}
	else if( numMatches < REGISTRATION_MIN_NUM_SURFELS ) {
		sumError = std::numeric_limits<double>::max();
		std::cout << "not enough surfels for robust matching " << numMatches << "\n";
		return false;
	}


	f = sumError / sumWeight;


	if( use_prior_pose_ ) {
		f += (prior_pose_mean_ - x).transpose() * prior_pose_invcov_ * (prior_pose_mean_ - x);
	}


	return true;

}



bool MultiResolutionColorSurfelRegistration::registrationErrorFunctionWithFirstAndSecondDerivativeLM( const Eigen::Matrix< double, 6, 1 >& x, MultiResolutionColorSurfelRegistration::RegistrationFunctionParameters& params, double& f, Eigen::Matrix< double, 6, 1 >& df, Eigen::Matrix< double, 6, 6 >& d2f, MultiResolutionColorSurfelRegistration::SurfelAssociationList& surfelAssociations ) {

	double sumError = 0.0;
	double sumWeight = 0.0;

	df.setZero();
	d2f.setZero();

	const double tx = x( 0 );
	const double ty = x( 1 );
	const double tz = x( 2 );
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	if( qx*qx+qy*qy+qz*qz > 1.0 )
		std::cout << "quaternion not stable!!\n";
	const double qw = params.lastWSign * sqrtf(1.0-qx*qx-qy*qy-qz*qz); // retrieve sign from last qw

	GradientFunctorLM gf( &surfelAssociations, tx, ty, tz, qx, qy, qz, qw, true );

	if( PARALLEL )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	int cidx = 0;
	if( params.correspondences_source_points_ ) {
		params.correspondences_source_points_->points.resize(surfelAssociations.size());
		params.correspondences_target_points_->points.resize(surfelAssociations.size());
	}


	double numMatches = 0;
	for( MultiResolutionColorSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;


		float nweight = it->n_src_->value_.assocWeight_ * it->n_dst_->value_.assocWeight_;
		float weight = nweight * it->weight;

		const Eigen::Matrix< double, 6, 3 > JtW = weight * it->df_dx.transpose() * it->W;

		df += JtW * (it->z - it->f);
		d2f += JtW * it->df_dx;

		sumError += weight * it->error;
		sumWeight += weight;
		numMatches += 1.0;//nweight;



		if( params.correspondences_source_points_ ) {

			pcl::PointXYZRGBA& p1 = params.correspondences_source_points_->points[cidx];
			pcl::PointXYZRGBA& p2 = params.correspondences_target_points_->points[cidx];

			Eigen::Vector4f pos1 = it->n_dst_->getCenterPosition();
			Eigen::Vector4f pos2 = it->n_src_->getCenterPosition();

			p1.x = pos1(0);
			p1.y = pos1(1);
			p1.z = pos1(2);

			p1.r = nweight * 255.f;
			p1.g = 0;
			p1.b = (1.f-nweight) * 255.f;

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = pos2.block<3,1>(0,0).cast<double>();
			pos(3,0) = 1.f;

			const Eigen::Vector4d pos_src = gf.currentTransform * pos;

			p2.x = pos_src[0];
			p2.y = pos_src[1];
			p2.z = pos_src[2];

			p2.r = nweight * 255.f;
			p2.g = 0;
			p2.b = (1.f-nweight) * 255.f;

			cidx++;
		}

	}


	if( params.correspondences_source_points_ ) {
		params.correspondences_source_points_->points.resize(cidx);
		params.correspondences_target_points_->points.resize(cidx);
	}

	if( sumWeight <= 1e-10 ) {
		sumError = std::numeric_limits<double>::max();
		return false;
	}
	else if( numMatches < REGISTRATION_MIN_NUM_SURFELS ) {
		sumError = std::numeric_limits<double>::max();
		std::cout << "not enough surfels for robust matching " << numMatches << "\n";
		return false;
	}

	f = sumError / sumWeight;
	df = df / sumWeight;
	d2f = d2f / sumWeight;


	if( use_prior_pose_ ) {

		f += (x - prior_pose_mean_).transpose() * prior_pose_invcov_ * (x - prior_pose_mean_);
		df += prior_pose_invcov_ * (prior_pose_mean_ - x);
		d2f += prior_pose_invcov_;

	}




	return true;

}



bool MultiResolutionColorSurfelRegistration::estimateTransformationNewton( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesTargetPoints, MultiResolutionColorSurfelRegistration::SurfelAssociationList* associations, int coarseToFineIterations, int fineIterations ) {

	Eigen::Matrix4d initialTransform = transform;

	// coarse alignment with features
	// fine alignment without features

	const int maxIterations = coarseToFineIterations + fineIterations;

	float minResolution = std::min( startResolution, stopResolution );
	float maxResolution = std::max( startResolution, stopResolution );

	const double step_max = 0.1;
	const double step_size_coarse = 1.0;
	const double step_size_fine = 1.0;

	Eigen::Matrix4d currentTransform = transform;

	// set up the minimization algorithm
	MultiResolutionColorSurfelRegistration::RegistrationFunctionParameters params;
	params.source = &source;
	params.target = &target;
	params.minResolution = minResolution;
	params.maxResolution = maxResolution;
	params.transform = &currentTransform;
	params.correspondences_source_points_ = correspondencesSourcePoints;
	params.correspondences_target_points_ = correspondencesTargetPoints;
	params.interpolate_neighbors = false;

	algorithm::OcTreeSamplingVectorMap<float, MultiResolutionColorSurfelMap::NodeValue> targetSamplingMap = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);
	params.targetSamplingMap = &targetSamplingMap;

	Eigen::Matrix< double, 6, 1 > x, last_x, df, best_x, best_g;
	Eigen::Matrix< double, 6, 6 > d2f;

	// initialize with current transform
	Eigen::Quaterniond q( currentTransform.block<3,3>(0,0) );

	x(0) = currentTransform( 0, 3 );
	x(1) = currentTransform( 1, 3 );
	x(2) = currentTransform( 2, 3 );
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	params.lastWSign = q.w() / fabsf(q.w());


	last_x = x;


	target.clearAssociations();

	double best_f = std::numeric_limits<double>::max();
	Eigen::Matrix4d bestTransform;
	bestTransform.setIdentity();
	best_x = x;
	best_g.setZero();


	pcl::StopWatch stopwatch;

	transform.setIdentity();
	const double qx = x( 3 );
	const double qy = x( 4 );
	const double qz = x( 5 );
	transform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( params.lastWSign*sqrt(1.0-qx*qx-qy*qy-qz*qz), qx, qy, qz ) );
	transform(0,3) = x( 0 );
	transform(1,3) = x( 1 );
	transform(2,3) = x( 2 );

	double associateTime = 0;
	double gradientTime = 0;

	MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;

	bool retVal = true;


	int iter = 0;
	while( iter < maxIterations ) {


		// stays at minresolution after coarseToFineIterations
		float searchDistFactor = 2.f;//std::max( 1.f, 1.f + 1.f * (((float)(fineIterations / 2 - iter)) / (float)(fineIterations / 2)) );
		float maxSearchDist = 2.f*maxResolution;//(minResolution + (maxResolution-minResolution) * ((float)(maxIterations - iter)) / (float)maxIterations);

		MultiResolutionColorSurfelRegistration::SurfelAssociationList tmpSurfelAssociations;
		MultiResolutionColorSurfelRegistration::SurfelAssociationList* surfelAssociations = associations;
		if( !surfelAssociations ) {
			surfelAssociations = &tmpSurfelAssociations;

			if( iter < coarseToFineIterations ) {
				surfelAssociations->clear();
				stopwatch.reset();
				associateMapsBreadthFirstParallel( *surfelAssociations, source, target, targetSamplingMap, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, true );
				double deltat = stopwatch.getTimeSeconds() * 1000.0;
				associateTime += deltat;
				params.interpolate_neighbors = false;

			}
			else {
				if( iter == coarseToFineIterations ) {
					target.clearAssociations();
				}

				surfelAssociations->clear();
				stopwatch.reset();
				associateMapsBreadthFirstParallel( *surfelAssociations, source, target, targetSamplingMap, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, false );
				double deltat = stopwatch.getTimeSeconds() * 1000.0;
				associateTime += deltat;
				params.interpolate_neighbors = true;
			}

		}


		// evaluate function and derivative
		double f = 0.0;
		stopwatch.reset();
		retVal = registrationErrorFunctionWithFirstAndSecondDerivative( x, true, (void*)&params, &f, df, d2f, *surfelAssociations );

		if( !retVal ) {
			df.setZero();
			d2f.setIdentity();
		}

		double deltat2 = stopwatch.getTimeSeconds() * 1000.0;
		gradientTime += deltat2;

		if( f < best_f ) {
			best_f = f;
			bestTransform = transform;
		}



		Eigen::Matrix< double, 6, 1 > lastX = x;
		Eigen::Matrix< double, 6, 6 > d2f_inv;
		d2f_inv.setZero();
		if( fabsf( d2f.determinant() ) > std::numeric_limits<double>::epsilon() ) {

			double step_size_i = step_size_fine;

			d2f_inv = d2f.inverse();
			Eigen::Matrix< double, 6, 1 > deltaX = -step_size_i * d2f_inv * df;

			last_x = x;


			double qx = x( 3 );
			double qy = x( 4 );
			double qz = x( 5 );
			double qw = params.lastWSign*sqrt(1.0-qx*qx-qy*qy-qz*qz);

			currentTransform.setIdentity();
			currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
			currentTransform(0,3) = x( 0 );
			currentTransform(1,3) = x( 1 );
			currentTransform(2,3) = x( 2 );


			qx = deltaX( 3 );
			qy = deltaX( 4 );
			qz = deltaX( 5 );
			qw = sqrt(1.0-qx*qx-qy*qy-qz*qz);

			Eigen::Matrix4d deltaTransform = Eigen::Matrix4d::Identity();
			deltaTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
			deltaTransform(0,3) = deltaX( 0 );
			deltaTransform(1,3) = deltaX( 1 );
			deltaTransform(2,3) = deltaX( 2 );

			Eigen::Matrix4d newTransform = deltaTransform * currentTransform;

			x( 0 ) = newTransform(0,3);
			x( 1 ) = newTransform(1,3);
			x( 2 ) = newTransform(2,3);

			Eigen::Quaterniond q_new( newTransform.block<3,3>(0,0) );
			x( 3 ) = q_new.x();
			x( 4 ) = q_new.y();
			x( 5 ) = q_new.z();

		}


		double qx = x( 3 );
		double qy = x( 4 );
		double qz = x( 5 );
		double qw = params.lastWSign*sqrt(1.0-qx*qx-qy*qy-qz*qz);



		if( isnan(qw) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f ) {
			x = last_x;
			return false;
		}


		transform.setIdentity();
		transform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		transform(0,3) = x( 0 );
		transform(1,3) = x( 1 );
		transform(2,3) = x( 2 );


		iter++;

	}


	return retVal;


}




bool MultiResolutionColorSurfelRegistration::estimateTransformationGradientDescent( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesTargetPoints, MultiResolutionColorSurfelRegistration::SurfelAssociationList* associations, int maxIterations ) {

	Eigen::Matrix4d initialTransform = transform;

	float minResolution = std::min( startResolution, stopResolution );
	float maxResolution = std::max( startResolution, stopResolution );

	// params for gradient descent
	const double step_max = 0.01;
	const double step_size_max = 0.1;
	const double step_size_start = 0.0001;
	const double step_size_min = 0.000001;

	Eigen::Matrix4d currentTransform = transform;

	// set up the minimization algorithm
	MultiResolutionColorSurfelRegistration::RegistrationFunctionParameters params;
	params.source = &source;
	params.target = &target;
	params.minResolution = minResolution;
	params.maxResolution = maxResolution;
	params.transform = &currentTransform;
	params.correspondences_source_points_ = correspondencesSourcePoints;
	params.correspondences_target_points_ = correspondencesTargetPoints;
	params.interpolate_neighbors = false;

	algorithm::OcTreeSamplingVectorMap<float, MultiResolutionColorSurfelMap::NodeValue> targetSamplingMap = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);
	params.targetSamplingMap = &targetSamplingMap;

	Eigen::Matrix< double, 6, 1 > x, df, best_x, best_g, last_x;
	Eigen::Matrix< double, 6, 6 > d2f;

	// initialize with current transform
	Eigen::Quaterniond q( currentTransform.block<3,3>(0,0) );

	x(0) = currentTransform( 0, 3 );
	x(1) = currentTransform( 1, 3 );
	x(2) = currentTransform( 2, 3 );
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	params.lastWSign = q.w() / fabsf(q.w());

	last_x = x;


	target.clearAssociations();


	double best_f = std::numeric_limits<double>::max();
	Eigen::Matrix4d bestTransform;
	bestTransform.setIdentity();
	best_x = x;
	best_g.setZero();

	bool retVal = true;

	double lastF = std::numeric_limits<double>::max();
	double step_size = step_size_start;

	pcl::StopWatch stopwatch;

	Eigen::Matrix< double, 6, 1 > g_im1;
	g_im1.setZero();


	int iter = 0;
	while( iter < maxIterations ) {

		float searchDistFactor = 2.f;
		float maxSearchDist = 2.f*maxResolution;

		MultiResolutionColorSurfelRegistration::SurfelAssociationList tmpSurfelAssociations;
		MultiResolutionColorSurfelRegistration::SurfelAssociationList* surfelAssociations = associations;
		if( !surfelAssociations ) {
			surfelAssociations = &tmpSurfelAssociations;
			surfelAssociations->clear();
			associateMapsBreadthFirstParallel( *surfelAssociations, source, target, targetSamplingMap, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, true );
		}


		// evaluate function and derivative
		double f = 0.0;
		params.interpolate_neighbors = false;
		retVal = registrationErrorFunctionWithFirstDerivative( x, (void*)&params, &f, df, *surfelAssociations );


		if( !retVal )
			df.setZero();

		Eigen::Matrix< double, 6, 1 > g_i = df;


		// use gradient direction, the likelihood may change when we add another resolution
		if( g_im1.dot( g_i ) < 0.f ) {
			step_size *= 0.5f;
		}
		else
			step_size *= 1.5f;

		step_size = std::min( step_size_max, std::max( step_size_min, step_size ) );


		// take a step in gradient direction
		double glen = g_i.norm();

		double gcap = 1.0;
		if( step_size * glen > step_max ) {
			gcap = step_max / (step_size * glen);
		}


		if( df.maxCoeff() > 1e100f || df.minCoeff() < -1e100f ) {
			std::cout << "x: " << x.transpose() << "\n";
			std::cout << "df: " << df.transpose() << "\n";
			std::cout << "g_i: " << g_i.transpose() << "\n";
			std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
		}

		x -= step_size * gcap * g_i;


		transform.setIdentity();
		const double qx = x( 3 );
		const double qy = x( 4 );
		const double qz = x( 5 );
		const double qw = params.lastWSign*sqrt(1.0-qx*qx-qy*qy-qz*qz);
		transform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		transform(0,3) = x( 0 );
		transform(1,3) = x( 1 );
		transform(2,3) = x( 2 );

		if( isnan( qw ) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f ) {
			transform = bestTransform;
			return false;
		}


		bestTransform = transform;

		lastF = f;
		g_im1 = g_i;


		iter++;
	}

	last_x = x;

	return retVal;

}



bool MultiResolutionColorSurfelRegistration::estimateTransformationLevenbergMarquardt( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesTargetPoints, int maxIterations ) {

	const bool useFeatures = true;

	const double tau = 10e-5;
	const double min_delta = 1e-3;
	const double min_error = 1e-6;

	Eigen::Matrix4d initialTransform = transform;

	float minResolution = std::min( startResolution, stopResolution );
	float maxResolution = std::max( startResolution, stopResolution );

	Eigen::Matrix4d currentTransform = transform;

	// set up the minimization algorithm
	MultiResolutionColorSurfelRegistration::RegistrationFunctionParameters params;
	params.source = &source;
	params.target = &target;
	params.minResolution = minResolution;
	params.maxResolution = maxResolution;
	params.transform = &currentTransform;
	params.correspondences_source_points_ = correspondencesSourcePoints;
	params.correspondences_target_points_ = correspondencesTargetPoints;

	algorithm::OcTreeSamplingVectorMap<float, MultiResolutionColorSurfelMap::NodeValue> targetSamplingMap = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);
	params.targetSamplingMap = &targetSamplingMap;


	// initialize with current transform
	Eigen::Matrix< double, 6, 1 > x;
	Eigen::Quaterniond q( currentTransform.block<3,3>(0,0) );

	x(0) = currentTransform( 0, 3 );
	x(1) = currentTransform( 1, 3 );
	x(2) = currentTransform( 2, 3 );
	x(3) = q.x();
	x(4) = q.y();
	x(5) = q.z();
	params.lastWSign = q.w() / fabsf(q.w());


	pcl::StopWatch stopwatch;

	Eigen::Matrix< double, 6, 1 > df;
	Eigen::Matrix< double, 6, 6 > d2f;

	const Eigen::Matrix< double, 6, 6 > id6 = Eigen::Matrix< double, 6, 6 >::Identity();
	double mu = -1.0;
	double nu = 2;

	double last_error = std::numeric_limits<double>::max();

	MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;

	bool reassociate = true;

	bool reevaluateGradient = true;

	bool retVal = true;

	int iter = 0;
	while( iter < maxIterations ) {

		if( reevaluateGradient ) {

			if( reassociate ) {
				target.clearAssociations();
			}

			float searchDistFactor = 2.f;
			float maxSearchDist = 2.f*maxResolution;

			stopwatch.reset();
			surfelAssociations.clear();
			associateMapsBreadthFirstParallel( surfelAssociations, source, target, targetSamplingMap, transform, 0.99f*minResolution, 1.01f*maxResolution, searchDistFactor, maxSearchDist, useFeatures );
			double deltat = stopwatch.getTime();

			stopwatch.reset();
			retVal = registrationErrorFunctionWithFirstAndSecondDerivativeLM( x, params, last_error, df, d2f, surfelAssociations );
			double deltat2 = stopwatch.getTime();
		}

		reevaluateGradient = false;

		if( !retVal ) {
			std::cout << "registration failed\n";
			return false;
		}

		if( mu < 0 ) {
			mu = tau * std::max( d2f.maxCoeff(), -d2f.minCoeff() );
		}

		Eigen::Matrix< double, 6, 1 > delta_x = Eigen::Matrix< double, 6, 1 >::Zero();
		Eigen::Matrix< double, 6, 6 > d2f_inv = Eigen::Matrix< double, 6, 6 >::Zero();
		if( fabsf( d2f.determinant() ) > std::numeric_limits<double>::epsilon() ) {

			d2f_inv = (d2f + mu * id6).inverse();

			delta_x = d2f_inv * df;

		}

		if( delta_x.norm() < min_delta ) {

			if( reassociate )
				break;

			reassociate = true;
			reevaluateGradient = true;
		}
		else
			reassociate = false;


		double qx = x( 3 );
		double qy = x( 4 );
		double qz = x( 5 );
		double qw = params.lastWSign*sqrt(1.0-qx*qx-qy*qy-qz*qz);


		currentTransform.setIdentity();
		currentTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		currentTransform(0,3) = x( 0 );
		currentTransform(1,3) = x( 1 );
		currentTransform(2,3) = x( 2 );


		qx = delta_x( 3 );
		qy = delta_x( 4 );
		qz = delta_x( 5 );
		qw = sqrt(1.0-qx*qx-qy*qy-qz*qz);

		Eigen::Matrix4d deltaTransform = Eigen::Matrix4d::Identity();
		deltaTransform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		deltaTransform(0,3) = delta_x( 0 );
		deltaTransform(1,3) = delta_x( 1 );
		deltaTransform(2,3) = delta_x( 2 );

		Eigen::Matrix4d newTransform = deltaTransform * currentTransform;

		Eigen::Matrix< double, 6, 1 > x_new;
		x_new( 0 ) = newTransform(0,3);
		x_new( 1 ) = newTransform(1,3);
		x_new( 2 ) = newTransform(2,3);

		Eigen::Quaterniond q_new( newTransform.block<3,3>(0,0) );
		x_new( 3 ) = q_new.x();
		x_new( 4 ) = q_new.y();
		x_new( 5 ) = q_new.z();


		double new_error = 0.0;
		bool retVal2 = registrationErrorFunctionLM( x_new, params, new_error, surfelAssociations );

		if( !retVal2 )
			return false;

		double rho = (last_error - new_error) / (delta_x.transpose() * (mu * delta_x + df));

		if( rho > 0 ) {

			x = x_new;

			mu *= std::max( 0.333, 1.0 - pow( 2.0*rho-1.0, 3.0 ) );
			nu = 2;

			reevaluateGradient = true;

		}
		else {

			mu *= nu; nu *= 2.0;

		}



		qx = x( 3 );
		qy = x( 4 );
		qz = x( 5 );
		qw = params.lastWSign*sqrt(1.0-qx*qx-qy*qy-qz*qz);



		if( isnan(qw) || fabsf(qx) > 1.f || fabsf(qy) > 1.f || fabsf(qz) > 1.f ) {
			return false;
		}


		transform.setIdentity();
		transform.block<3,3>(0,0) = Eigen::Matrix3d( Eigen::Quaterniond( qw, qx, qy, qz ) );
		transform(0,3) = x( 0 );
		transform(1,3) = x( 1 );
		transform(2,3) = x( 2 );


//		last_error = new_error;

		iter++;

	}


	return retVal;

}

bool MultiResolutionColorSurfelRegistration::estimateTransformation( MultiResolutionColorSurfelMap& source, const boost::shared_ptr< const pcl::PointCloud<pcl::PointXYZRGBA> >& cloud, const boost::shared_ptr< const std::vector< int > >& indices, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesTargetPoints, int gradientIterations, int coarseToFineIterations, int fineIterations, MultiResolutionColorSurfelRegistration::SurfelAssociationList* associations ) {
	return estimateTransformation( source, *cloud, *indices, transform, startResolution, stopResolution, correspondencesSourcePoints, correspondencesTargetPoints, gradientIterations, coarseToFineIterations, fineIterations, associations );
}

bool MultiResolutionColorSurfelRegistration::estimateTransformation( MultiResolutionColorSurfelMap& source, const pcl::PointCloud<pcl::PointXYZRGBA>& cloud, const std::vector< int >& indices, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesTargetPoints, int gradientIterations, int coarseToFineIterations, int fineIterations, MultiResolutionColorSurfelRegistration::SurfelAssociationList* associations ) {

	// add points to local map
	MultiResolutionColorSurfelMap target( source.min_resolution_, source.max_range_ );
	target.addPoints( cloud, indices );
	target.evaluateSurfels();

	// estimate transformation from maps
	return estimateTransformation( source, target, transform, startResolution, stopResolution, correspondencesSourcePoints, correspondencesTargetPoints, gradientIterations, coarseToFineIterations, fineIterations, associations );

}

bool MultiResolutionColorSurfelRegistration::estimateTransformation( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZRGBA >::Ptr correspondencesTargetPoints, int gradientIterations, int coarseToFineIterations, int fineIterations, MultiResolutionColorSurfelRegistration::SurfelAssociationList* associations ) {

	// estimate transformation from maps
	target.clearAssociations();

	bool retVal = true;
	if( gradientIterations > 0 )
		retVal = estimateTransformationLevenbergMarquardt( source, target, transform, startResolution, stopResolution, correspondencesSourcePoints, correspondencesTargetPoints, gradientIterations );

	if( !retVal )
		std::cout << "levenberg marquardt failed\n";

	Eigen::Matrix4d transformGradient = transform;

	if( retVal ) {

		bool retVal2 = estimateTransformationNewton( source, target, transform, startResolution, stopResolution, correspondencesSourcePoints, correspondencesTargetPoints, associations, coarseToFineIterations, fineIterations );
		if( !retVal2 ) {
			std::cout << "newton failed\n";
			transform = transformGradient;

			if( gradientIterations == 0 )
				retVal = false;
		}

	}

	return retVal;

}


// transform: transforms source to target
// intended to have the "smaller" map (the model) in target
double MultiResolutionColorSurfelRegistration::matchLogLikelihood( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform ) {

	double sumLogLikelihood = 0.0;

	const double normalStd = 0.125*M_PI;
	const double normalMinLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd ) - 8.0;

	Eigen::Matrix4d targetToSourceTransform = transform.inverse();
	Eigen::Matrix3d currentRotation = Eigen::Matrix3d( targetToSourceTransform.block<3,3>(0,0) );
	Eigen::Matrix3d currentRotationT = currentRotation.transpose();
	Eigen::Vector3d currentTranslation = Eigen::Vector3d( targetToSourceTransform.block<3,1>(0,3) );

	// start at highest resolution in the tree and compare recursively
	MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;
	std::list< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* > openNodes;
	openNodes.push_back( target.octree_->root_ );
	while( !openNodes.empty() ) {
		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n = openNodes.front();
		openNodes.pop_front();

		for( unsigned int i = 0; i < 8; i++ ) {
			if( n->children_[i] )
				openNodes.push_back( n->children_[i] );
		}

		const float processResolution = n->resolution();

		Eigen::Vector4d npos = n->getPosition().cast<double>();
		Eigen::Vector4d npos_match_src = targetToSourceTransform * npos;

		// for match log likelihood: query in volume to check the neighborhood for the best matching (discretization issues)
		std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* > nodes;
		nodes.reserve(50);
		const double searchRadius = 2.0 * n->resolution();
		Eigen::Vector4f minPosition, maxPosition;
		minPosition[0] = npos_match_src(0) - searchRadius;
		minPosition[1] = npos_match_src(1) - searchRadius;
		minPosition[2] = npos_match_src(2) - searchRadius;
		maxPosition[0] = npos_match_src(0) + searchRadius;
		maxPosition[1] = npos_match_src(1) + searchRadius;
		maxPosition[2] = npos_match_src(2) + searchRadius;
		source.octree_->getAllNodesInVolumeOnDepth( nodes, minPosition, maxPosition, n->depth_, false );

		Eigen::Matrix3d cov_add;
		cov_add.setZero();
		if( ADD_SMOOTH_POS_COVARIANCE ) {
			cov_add.setIdentity();
			cov_add *= SMOOTH_SURFACE_COV_FACTOR * processResolution*processResolution;
		}


		// only consider model surfels that are visible from the scene viewpoint under the given transformation

		for( unsigned int i = 0; i < MAX_NUM_SURFELS; i++ ) {

			MultiResolutionColorSurfelMap::Surfel* modelSurfel = &n->value_.surfels_[i];

			if( modelSurfel->num_points_ < MIN_SURFEL_POINTS ) {
				continue;
			}

			// transform surfel mean with current transform and find corresponding node in source for current resolution
			// find corresponding surfel in node via the transformed view direction of the surfel

			Eigen::Vector4d pos;
			pos.block<3,1>(0,0) = modelSurfel->mean_.block<3,1>(0,0);
			pos(3,0) = 1.f;

			Eigen::Vector4d dir;
			dir.block<3,1>(0,0) = modelSurfel->initial_view_dir_;
			dir(3,0) = 0.f; // pure rotation

			Eigen::Vector4d pos_match_src = targetToSourceTransform * pos;
			Eigen::Vector4d dir_match_src = targetToSourceTransform * dir;

			// precalculate log likelihood when surfel is not matched in the scene
			Eigen::Matrix3d cov2 = modelSurfel->cov_.block<3,3>(0,0);
			cov2 += cov_add;

			Eigen::Matrix3d cov2_RT = cov2 * currentRotationT;
			Eigen::Matrix3d cov2_rotated = (currentRotation * cov2_RT).eval();

			double nomatch_loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov2_rotated.determinant() ) - 10.0 * 24.0;

			nomatch_loglikelihood += normalMinLogLikelihood;

			if( std::isinf<double>(nomatch_loglikelihood) || std::isnan<double>(nomatch_loglikelihood) )
				continue;

			double bestSurfelLogLikelihood = nomatch_loglikelihood;

			// is model surfel visible from the scene viewpoint?
			// assumption: scene viewpoint in (0,0,0)
			if( dir_match_src.block<3,1>(0,0).dot( pos_match_src.block<3,1>(0,0) / pos_match_src.block<3,1>(0,0).norm() ) < cos(0.25*M_PI) ) {
				sumLogLikelihood += bestSurfelLogLikelihood;
				continue;
			}

			for( std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >::iterator it = nodes.begin(); it != nodes.end(); ++it ) {

				spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src = *it;

				// find best matching surfel for the view direction in the scene map
				MultiResolutionColorSurfelMap::Surfel* bestMatchSurfel = NULL;
				double bestMatchDist = -1.f;
				for( unsigned int k = 0; k < MAX_NUM_SURFELS; k++ ) {

					const double dist = dir_match_src.block<3,1>(0,0).dot( n_src->value_.surfels_[k].initial_view_dir_ );
					if( dist > bestMatchDist ) {
						bestMatchSurfel = &n_src->value_.surfels_[k];
						bestMatchDist = dist;
					}
				}


				// do only associate on the same resolution
				// no match? use maximum distance log likelihood for this surfel
				if( bestMatchSurfel->num_points_ < MIN_SURFEL_POINTS ) {
					continue;
				}


				Eigen::Vector3d diff_pos = bestMatchSurfel->mean_.block<3,1>(0,0) - pos_match_src.block<3,1>(0,0);


				Eigen::Matrix3d cov1 = bestMatchSurfel->cov_.block<3,3>(0,0);
				cov1 += cov_add;

				Eigen::Matrix3d cov = cov1 + cov2_rotated;
				Eigen::Matrix3d invcov = cov.inverse().eval();

				double exponent = -0.5 * diff_pos.dot(invcov * diff_pos);
				exponent = std::max( -24.0, exponent ); // -32: -0.5 * ( 16 + 16 + 16 + 16 )
				double loglikelihood = -0.5 * log( 8.0 * M_PI * M_PI * M_PI * cov.determinant() ) + exponent;


				// test: also consider normal orientation in the likelihood!!
				Eigen::Vector4d normal_src;
				normal_src.block<3,1>(0,0) = modelSurfel->normal_;
				normal_src(3,0) = 0.0;
				normal_src = (targetToSourceTransform * normal_src).eval();

				double normalError = std::min( 4.0 * normalStd, acos( normal_src.block<3,1>(0,0).dot( bestMatchSurfel->normal_ ) ) );
				double normalExponent = -0.5 * normalError * normalError / ( normalStd*normalStd );
				double normalLogLikelihood = -0.5 * log( 2.0 * M_PI * normalStd ) + normalExponent;


				if( std::isinf<double>(nomatch_loglikelihood) || std::isnan<double>( exponent ) ) {
					continue;
				}
				if( std::isinf<double>(nomatch_loglikelihood) || std::isnan<double>(loglikelihood) )
					continue;

				bestSurfelLogLikelihood = std::max( bestSurfelLogLikelihood, loglikelihood + normalLogLikelihood );

			}

			sumLogLikelihood += bestSurfelLogLikelihood;
		}


	}

	return sumLogLikelihood;


}



bool MultiResolutionColorSurfelRegistration::estimatePoseCovariance( Eigen::Matrix< double, 6, 6 >& poseCov, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution ) {

	target.clearAssociations();

	float minResolution = std::min( startResolution, stopResolution );
	float maxResolution = std::max( startResolution, stopResolution );

	algorithm::OcTreeSamplingVectorMap<float, MultiResolutionColorSurfelMap::NodeValue> targetSamplingMap = algorithm::downsampleVectorOcTree(*target.octree_, false, target.octree_->max_depth_);

	double sumWeight = 0.0;

	Eigen::Quaterniond q( transform.block<3,3>(0,0) );

	const double tx = transform(0,3);
	const double ty = transform(1,3);
	const double tz = transform(2,3);
	const double qx = q.x();
	const double qy = q.y();
	const double qz = q.z();
	const double qw = q.w();


	MultiResolutionColorSurfelRegistration::SurfelAssociationList surfelAssociations;
	associateMapsBreadthFirstParallel( surfelAssociations, source, target, targetSamplingMap, transform, 0.99f*minResolution, 1.01f*maxResolution, 2.f, 2.f*maxResolution, false );


	GradientFunctor gf( &surfelAssociations, tx, ty, tz, qx, qy, qz, qw, false, true, true, true );

	if( PARALLEL )
		tbb::parallel_for( tbb::blocked_range<size_t>(0,surfelAssociations.size()), gf );
	else
		std::for_each( surfelAssociations.begin(), surfelAssociations.end(), gf );

	Eigen::Matrix< double, 6, 6 > d2f, JSzJ;
	d2f.setZero();
	JSzJ.setZero();

	for( MultiResolutionColorSurfelRegistration::SurfelAssociationList::iterator it = surfelAssociations.begin(); it != surfelAssociations.end(); ++it ) {

		if( !it->match )
			continue;

		d2f += it->weight * it->d2f;
		JSzJ += it->weight * it->JSzJ;
		sumWeight += it->weight;

	}


	if( sumWeight <= 1e-10 ) {
		poseCov.setIdentity();
		return false;
	}
	else if( sumWeight < REGISTRATION_MIN_NUM_SURFELS ) {
		std::cout << "not enough surfels for robust matching\n";
		poseCov.setIdentity();
		return false;
	}
	else {
		d2f /= sumWeight;
		JSzJ /= sumWeight;
	}

	poseCov.setZero();

	if( fabsf(d2f.determinant()) < 1e-8 ) {
		poseCov.setIdentity();
		return false;
	}

	poseCov = d2f.inverse() * JSzJ * d2f.inverse();

	return true;


}



