#include "eg_feeder/eg_feeder_node.h"

using namespace std;
using namespace Chakra;

shared_ptr<ChakraProtoMsg::Node> EGFeederNode::getChakraNode() {
  return node_;
}

void EGFeederNode::setChakraNode(shared_ptr<ChakraProtoMsg::Node> node) {
  node_ = node;
}

void EGFeederNode::addChild(shared_ptr<EGFeederNode> node) {
  // Avoid adding the same child node multiple times
  // addChild is called multiple times to resolve dependencies
  if (children_set_.find(node) != children_set_.end()) {
    return;
  }
  children_vec_.emplace_back(node);
  children_set_.emplace(node);
}

vector<shared_ptr<EGFeederNode>> EGFeederNode::getChildren() {
  return children_vec_;
}

void EGFeederNode::addDepUnresolvedParentID(uint64_t node_id) {
  dep_unresolved_parent_ids_.emplace_back(node_id);
}

vector<uint64_t> EGFeederNode::getDepUnresolvedParentids_() {
  return dep_unresolved_parent_ids_;
}

void EGFeederNode::setDepUnresolvedParentIDs(
    vector<uint64_t> const& dep_unresolved_parent_ids) {
  dep_unresolved_parent_ids_ = dep_unresolved_parent_ids;
}
