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
  for (auto child: children_) {
    if (child == node)
      return;
  }
  children_.push_back(node);
}

vector<shared_ptr<EGFeederNode>> EGFeederNode::getChildren() {
  return children_;
}

void EGFeederNode::addDepUnresolvedParentID(uint64_t node_id) {
  dep_unresolved_parent_id_.push_back(node_id);
}

vector<uint64_t> EGFeederNode::getDepUnresolvedParentID() {
  return dep_unresolved_parent_id_;
}

void EGFeederNode::setDepUnresolvedParentID(
    vector<uint64_t> dep_unresolved_parent_id) {
  dep_unresolved_parent_id_.clear();
  for (auto id: dep_unresolved_parent_id) {
    dep_unresolved_parent_id_.push_back(id);
  }
}
