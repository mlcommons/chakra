#include "et_feeder/et_feeder_node.h"

using namespace std;
using namespace Chakra;

shared_ptr<ChakraProtoMsg::Node> ETFeederNode::getChakraNode() {
  return node_;
}

void ETFeederNode::setChakraNode(shared_ptr<ChakraProtoMsg::Node> node) {
  node_ = node;
}

void ETFeederNode::addChild(shared_ptr<ETFeederNode> node) {
  // Avoid adding the same child node multiple times
  // addChild is called multiple times to resolve dependencies
  if (children_set_.find(node) != children_set_.end()) {
    return;
  }
  children_vec_.emplace_back(node);
  children_set_.emplace(node);
}

vector<shared_ptr<ETFeederNode>> ETFeederNode::getChildren() {
  return children_vec_;
}

void ETFeederNode::addDepUnresolvedParentID(uint64_t node_id) {
  dep_unresolved_parent_ids_.emplace_back(node_id);
}

vector<uint64_t> ETFeederNode::getDepUnresolvedParentIDs() {
  return dep_unresolved_parent_ids_;
}

void ETFeederNode::setDepUnresolvedParentIDs(
    vector<uint64_t> const& dep_unresolved_parent_ids) {
  dep_unresolved_parent_ids_ = dep_unresolved_parent_ids;
}
