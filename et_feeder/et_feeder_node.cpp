#include "et_feeder/et_feeder_node.h"

using namespace std;
using namespace Chakra;

ETFeederNode::ETFeederNode(std::shared_ptr<ChakraProtoMsg::Node> node) {
  this->node_ = node;
  this->id_ = node->id();
  this->name_ = node->name();
  this->runtime_ = node->duration_micros();
  this->is_cpu_op_ = 1;

  for (const auto& attr : node->attr()) {
    const string& attr_name = attr.name();

    if (attr_name == "is_cpu_op") {
      this->is_cpu_op_ = static_cast<uint32_t>(attr.int32_val());
    } else if (attr_name == "num_ops") {
      this->num_ops_ = static_cast<uint64_t>(attr.int64_val());
    } else if (attr_name == "tensor_size") {
      this->tensor_size_ = attr.uint64_val();
    } else if (attr_name == "comm_type") {
      this->comm_type_ =
          static_cast<ChakraProtoMsg::CollectiveCommType>(attr.int64_val());
    } else if (attr_name == "involved_dim") {
      this->involved_dim_.clear();
      for (const bool val : attr.bool_list().values()) {
        this->involved_dim_.push_back(val);
      }
      this->involved_dim_size_ = this->involved_dim_.size();
    } else if (attr_name == "comm_priority") {
      this->comm_priority_ = static_cast<uint32_t>(attr.int32_val());
    } else if (attr_name == "comm_size") {
      this->comm_size_ = attr.int64_val();
    } else if (attr_name == "comm_src") {
      this->comm_src_ = static_cast<uint32_t>(attr.int32_val());
    } else if (attr_name == "comm_dst") {
      this->comm_dst_ = static_cast<uint32_t>(attr.int32_val());
    } else if (attr_name == "comm_tag") {
      this->comm_tag_ = static_cast<uint32_t>(attr.int32_val());
    }
  }
}

shared_ptr<ChakraProtoMsg::Node> ETFeederNode::getChakraNode() {
  return node_;
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

uint64_t ETFeederNode::id() {
  return id_;
}

string ETFeederNode::name() {
  return name_;
}

bool ETFeederNode::is_cpu_op() {
  return is_cpu_op_;
}

ChakraProtoMsg::NodeType ETFeederNode::type() {
  return node_->type();
}

uint64_t ETFeederNode::runtime() {
  return runtime_;
}

uint64_t ETFeederNode::num_ops() {
  return num_ops_;
}

uint32_t ETFeederNode::tensor_loc() {
  return tensor_loc_;
}

uint64_t ETFeederNode::tensor_size() {
  return tensor_size_;
}

ChakraProtoMsg::CollectiveCommType ETFeederNode::comm_type() {
  return comm_type_;
}

uint32_t ETFeederNode::involved_dim_size() {
  return involved_dim_size_;
}

bool ETFeederNode::involved_dim(int i) {
  return involved_dim_[i];
}

uint32_t ETFeederNode::comm_priority() {
  return comm_priority_;
}

uint64_t ETFeederNode::comm_size() {
  return comm_size_;
}

uint32_t ETFeederNode::comm_src() {
  return comm_src_;
}

uint32_t ETFeederNode::comm_dst() {
  return comm_dst_;
}

uint32_t ETFeederNode::comm_tag() {
  return comm_tag_;
}
