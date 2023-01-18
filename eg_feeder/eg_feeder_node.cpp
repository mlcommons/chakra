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
  children_.push_back(node);
}

vector<shared_ptr<EGFeederNode>> EGFeederNode::getChildren() {
  return children_;
}
