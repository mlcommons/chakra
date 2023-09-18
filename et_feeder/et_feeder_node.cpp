#include "et_feeder/et_feeder_node.h"

using namespace std;
using namespace Chakra;

ETFeederNode::ETFeederNode(std::shared_ptr<ChakraProtoMsg::Node> node) {
  this->node_= node;
  this->id_ = node->id();
  this->name_ = node->name();
  this->runtime_ = node->duration_micros();
  this->is_cpu_op_ = true;
  for (int i = 0; i < node->attr_size(); i++) {
    string attr_name = node->attr(i).name();
    if (attr_name == "is_cpu_op") {
      assign_attr_val(node, i, (void *)(&is_cpu_op_));
    } else if (attr_name == "num_ops") {
      assign_attr_val(node, i, (void *)(&num_ops_));
    } else if (attr_name == "tensor_size") {
      assign_attr_val(node, i, (void *)(&tensor_size_));
    } else if (attr_name == "comm_type") {
      assign_attr_val(node, i, (void *)(&comm_type_));
    } else if (attr_name == "involved_dim") {
      assign_attr_val(node, i, (void *)(&involved_dim_));
      involved_dim_size_ = node->attr(i).bool_list().values_size();
    } else if (attr_name == "comm_priority") {
      assign_attr_val(node, i, (void *)(&comm_priority_));
    } else if (attr_name == "comm_size") {
      assign_attr_val(node, i, (void *)(&comm_size_));
    } else if (attr_name == "comm_src") {
      assign_attr_val(node, i, (void *)(&comm_src_));
    } else if (attr_name == "comm_dst") {
      assign_attr_val(node, i, (void *)(&comm_dst_));
    } else if (attr_name == "comm_tag") {
      assign_attr_val(node, i, (void *)(&comm_tag_));
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

void ETFeederNode::assign_attr_val(shared_ptr<ChakraProtoMsg::Node> node, int i, void *member) {
  auto attr = node->attr(i);
  switch(attr.value_case()) {
    case ChakraProtoMsg::AttributeProto::kDoubleVal:
      *((double *)member) = attr.double_val();
      break;
    case ChakraProtoMsg::AttributeProto::kDoubleList:
      for (const auto& val : attr.double_list().values()) {
        (*((std::vector<double> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kFloatVal:
      *((float *)member) = attr.float_val();
      break;
    case ChakraProtoMsg::AttributeProto::kFloatList:
      for (const auto& val : attr.float_list().values()) {
        (*((std::vector<float> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kInt32Val:
      *((int32_t *)member) = attr.int32_val();
      break;
    case ChakraProtoMsg::AttributeProto::kInt32List:
      for (const auto& val : attr.int32_list().values()) {
        (*((std::vector<int32_t> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kInt64Val:
      *((int64_t *)member) = attr.int64_val();
      break;
    case ChakraProtoMsg::AttributeProto::kInt64List:
      for (const auto& val : attr.int64_list().values()) {
        (*((std::vector<int64_t> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kUint32Val:
      *((uint32_t *)member) = attr.uint32_val();
      break;
    case ChakraProtoMsg::AttributeProto::kUint32List:
      for (const auto& val : attr.uint32_list().values()) {
        (*((std::vector<uint32_t> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kUint64Val:
      *((uint64_t *)member) = attr.uint64_val();
      break;
    case ChakraProtoMsg::AttributeProto::kUint64List:
      for (const auto& val : attr.uint64_list().values()) {
        (*((std::vector<uint64_t> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kSint32Val:
      *((int32_t *)member) = attr.sint32_val();
      break;
    case ChakraProtoMsg::AttributeProto::kSint32List:
      for (const auto& val : attr.sint32_list().values()) {
        (*((std::vector<int32_t> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kSint64Val:
      *((int64_t *)member) = attr.sint64_val();
      break;
    case ChakraProtoMsg::AttributeProto::kSint64List:
      for (const auto& val : attr.sint64_list().values()) {
        (*((std::vector<int64_t> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kFixed32Val:
      *((uint32_t *)member) = attr.fixed32_val();
      break;
    case ChakraProtoMsg::AttributeProto::kFixed32List:
      for (const auto& val : attr.fixed32_list().values()) {
        (*((std::vector<uint32_t> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kFixed64Val:
      *((uint64_t *)member) = attr.fixed64_val();
      break;
    case ChakraProtoMsg::AttributeProto::kFixed64List:
      for (const auto& val : attr.fixed64_list().values()) {
        (*((std::vector<uint64_t> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kSfixed32Val:
      *((int32_t *)member) = attr.sfixed32_val();
      break;
    case ChakraProtoMsg::AttributeProto::kSfixed32List:
      for (const auto& val : attr.sfixed32_list().values()) {
        (*((std::vector<int32_t> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kSfixed64Val:
      *((int64_t *)member) = attr.sfixed64_val();
      break;
    case ChakraProtoMsg::AttributeProto::kSfixed64List:
      for (const auto& val : attr.sfixed64_list().values()) {
        (*((std::vector<int64_t> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kBoolVal:
      *((bool *)member) = attr.bool_val();
      break;
    case ChakraProtoMsg::AttributeProto::kBoolList:
      for (const auto& val : attr.bool_list().values()) {
        (*((std::vector<bool> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kStringVal:
      *((std::string *)member) = attr.string_val();
      break;
    case ChakraProtoMsg::AttributeProto::kStringList:
      for (const auto& val : attr.string_list().values()) {
        (*((std::vector<std::string> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::kBytesVal:
      *((std::string *)member) = attr.bytes_val();
      break;
    case ChakraProtoMsg::AttributeProto::kBytesList:
      for (const auto& val : attr.bytes_list().values()) {
        (*((std::vector<std::string> *)member)).push_back(val);
      }
      break;
    case ChakraProtoMsg::AttributeProto::VALUE_NOT_SET:
    default:
      std::cerr << "undefined attr type in chakra node" << std::endl;
      exit(EXIT_FAILURE);
      break;
  }
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
