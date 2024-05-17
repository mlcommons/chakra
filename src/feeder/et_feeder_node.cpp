#include "et_feeder_node.h"

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
      this->is_cpu_op_ = static_cast<bool>(attr.bool_val());
    } else if (attr_name == "num_ops") {
      this->num_ops_ = static_cast<uint64_t>(attr.int64_val());
    } else if (attr_name == "tensor_size") {
      this->tensor_size_ = attr.uint64_val();
    } else if (attr_name == "comm_type") {
      this->comm_type_ =
          static_cast<ChakraProtoMsg::CollectiveCommType>(attr.int64_val());
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
    } else {
      this->other_attrs_.emplace(attr_name, attr);
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

const ChakraProtoMsg::AttributeProto& ETFeederNode::get_other_attr(
    const string& attr_name) const {
  if (this->has_other_attr(attr_name))
    return this->other_attrs_.at(attr_name);
  throw std::runtime_error(
      "Asked for attr \"" + attr_name + "\" from node " +
      std::to_string(this->id_) + ", which do not exist");
}

bool ETFeederNode::has_other_attr(const string& attr_name) const {
  const auto& item = this->other_attrs_.find(attr_name);
  return item != this->other_attrs_.end();
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
