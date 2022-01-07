/* ----------------------------------------------------------------------------
 * Copyright 2021 The Ambitious Folks of the MRG
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   HybridFactorGraph.cpp
 * @brief  Custom hybrid factor graph for discrete + continuous factors
 * @author Kevin Doherty, kdoherty@mit.edu
 * @date   December 2021
 */

#include <gtsam/hybrid/DCGaussianMixtureFactor.h>
#include <gtsam/hybrid/HybridEliminationTree.h>
#include <gtsam/hybrid/HybridFactorGraph.h>
#include <gtsam/inference/EliminateableFactorGraph-inst.h>
#include <gtsam/linear/HessianFactor.h>

#include <boost/make_shared.hpp>

using namespace std;

namespace gtsam {

// Instantiate base classes
// template class FactorGraph<Factor>;
template class EliminateableFactorGraph<HybridFactorGraph>;

void HybridFactorGraph::print(const string& str,
                              const gtsam::KeyFormatter& keyFormatter) const {
  string prefix = str.empty() ? str : str + ".";
  cout << prefix << "size: " << size() << endl;
  nonlinearGraph_.print(prefix + "NonlinearFactorGraph", keyFormatter);
  discreteGraph_.print(prefix + "DiscreteFactorGraph", keyFormatter);
  dcGraph_.print(prefix + "DCFactorGraph", keyFormatter);
}

GaussianHybridFactorGraph HybridFactorGraph::linearize(
    const Values& continuousValues) const {
  // linearize the continuous factors
  auto gaussianFactorGraph = nonlinearGraph_.linearize(continuousValues);

  // linearize the DCFactors
  DCFactorGraph linearized_DC_factors;
  for (auto&& dcFactor : dcGraph_) {
    // If dcFactor is a DCGaussianMixtureFactor, we don't linearize.
    if (boost::dynamic_pointer_cast<DCGaussianMixtureFactor>(dcFactor)) {
      linearized_DC_factors.push_back(dcFactor);
    } else {
      auto linearizedDCFactor = dcFactor->linearize(continuousValues);
      linearized_DC_factors.push_back(linearizedDCFactor);
    }
  }

  // Construct new GaussianHybridFactorGraph
  return GaussianHybridFactorGraph(*gaussianFactorGraph, discreteGraph_,
                                   linearized_DC_factors);
}

bool HybridFactorGraph::equals(const HybridFactorGraph& other,
                               double tol) const {
  return Base::equals(other, tol) &&
         nonlinearGraph_.equals(other.nonlinearGraph_, tol) &&
         discreteGraph_.equals(other.discreteGraph_, tol) &&
         dcGraph_.equals(other.dcGraph_, tol);
}

void HybridFactorGraph::clear() {
  nonlinearGraph_.resize(0);
  discreteGraph_.resize(0);
  dcGraph_.resize(0);
}

/// Define adding a GaussianFactor to a sum.
using Sum = DCGaussianMixtureFactor::Sum;
static Sum& operator+=(Sum& sum, const GaussianFactor::shared_ptr& factor) {
  using Y = GaussianFactorGraph;
  auto add = [&factor](const Y& graph) {
    auto result = graph;
    result.push_back(factor);
    return result;
  };
  sum = sum.apply(add);
  return sum;
}

ostream& operator<<(ostream& os,
                    const GaussianFactorGraph::EliminationResult& er) {
  os << "ER" << endl;
  return os;
}

// The function type that does a single elimination step on a variable.
pair<GaussianMixture::shared_ptr, boost::shared_ptr<Factor>> EliminateHybrid(
    const GaussianHybridFactorGraph& factors, const Ordering& ordering) {
  // STEP 1: SUM
  // Create a new decision tree with all factors gathered at leaves.
  Sum sum = factors.sum();

  // STEP 1: ELIMINATE
  // Eliminate each sum using conventional Cholesky:
  // We can use this by creating a *new* decision tree:
  using GFG = GaussianFactorGraph;
  using Pair = GaussianFactorGraph::EliminationResult;

  KeyVector keys;
  KeyVector separatorKeys;  // Do with optional?
  auto eliminate = [&](const GFG& graph) {
    auto result = EliminatePreferCholesky(graph, ordering);
    if (keys.size() == 0) keys = result.first->keys();
    if (separatorKeys.size() == 0) separatorKeys = result.second->keys();
    return result;
  };
  DecisionTree<Key, Pair> eliminationResults(sum, eliminate);

  // STEP 3: Create result
  // TODO(Frank): auto pair = eliminationResults.unzip();

  const DiscreteKeys discreteKeys = factors.discreteKeys();

  // Grab the conditionals and create the GaussianMixture
  auto first = [](const Pair& result) { return result.first; };
  GaussianMixture::Conditionals conditionals(eliminationResults, first);
  auto conditional =
      boost::make_shared<GaussianMixture>(keys, discreteKeys, conditionals);

  // If there are no more continuous parents, then we should create here a
  // DiscreteFactor, with the error for each discrete choice.
  if (separatorKeys.size() == 0) {
    auto discreteFactor = HybridFactorGraph::toDecisionFactor(factors);
    return {conditional, discreteFactor};

  } else {
    // Create a resulting DCGaussianMixture on the separator.
    auto second = [](const Pair& result) { return result.second; };
    DCGaussianMixtureFactor::Factors separatorFactors(eliminationResults,
                                                      second);
    auto factor = boost::make_shared<DCGaussianMixtureFactor>(
        separatorKeys, discreteKeys, separatorFactors);
    return {conditional, factor};
  }
}

DecisionTreeFactor::shared_ptr HybridFactorGraph::toDecisionTreeFactor(
    const GaussianHybridFactorGraph& ghfg) {
  using GFG = GaussianFactorGraph;

  Sum sum = ghfg.sum();

  // Get the decision tree with each leaf as the error for that assignment
  std::function<double(GaussianFactorGraph)> gfgError = [&](const GFG& graph) {
    VectorValues values = graph.optimize();
    return graph.error(values);
  };
  DecisionTree<Key, double> gfgdt(sum, gfgError);

  auto allAssignments = cartesianProduct<Key>(discreteKeys);
  sum(allAssignments[0]).print("GFG 1");
  std::cout << "=======================" << std::endl;
  sum(allAssignments[1]).print("GFG 2");
  std::cout << "=======================" << std::endl;
  sum(allAssignments[2]).print("GFG 3");
  std::cout << "=======================" << std::endl;
  sum(allAssignments[3]).print("GFG 4");
  std::cout << "=======================" << std::endl;

  DecisionTree<Key, double>::ValueFormatter valueFormatter =
      [](const double& error) {
        stringstream ss;
        ss << error;
        return ss.str();
      };

  auto factor = boost::make_shared<DecisionTreeFactor>(discreteKeys, gfgdt);
  return factor;
}

}  // namespace gtsam
