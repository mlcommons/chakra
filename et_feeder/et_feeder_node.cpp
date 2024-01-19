#include "et_feeder/et_feeder_node.h"

#include <exception>

using namespace std;
using namespace Chakra;

ETFeederNode::ETFeederNode(std::shared_ptr<ChakraProtoMsg::Node> node) {
  this->node_ = node;
  this->id_ = node->id();
  this->name_ = node->name();
  this->runtime_.emplace(node->duration_micros());

  for (int i = 0; i < node->attr_size(); i++) {
    string attr_name = node->attr(i).name();
    try {
      if (attr_name == "is_cpu_op") {
        assign_attr_val(node->attr(i), static_cast<any>(&is_cpu_op_));
      } else if (attr_name == "num_ops") {
        assign_attr_val(node->attr(i), static_cast<any>(&num_ops_));
      } else if (attr_name == "tensor_size") {
        assign_attr_val(node->attr(i), static_cast<any>(&tensor_size_));
      } else if (attr_name == "comm_type") {
        // TODO: no type of attr fields for comm_type, and it is stored in
        // int64_val()
        comm_type_.emplace(static_cast<ChakraProtoMsg::CollectiveCommType>(
            node->attr(i).int64_val()));
      } else if (attr_name == "involved_dim") {
        assign_attr_val(node->attr(i), static_cast<any>(&involved_dim_));
        this->involved_dim_size_.emplace(
            node->attr(i).bool_list().values_size());
      } else if (attr_name == "comm_priority") {
        assign_attr_val(node->attr(i), static_cast<any>(&comm_priority_));
      } else if (attr_name == "comm_size") {
        assign_attr_val(node->attr(i), static_cast<any>(&comm_size_));
      } else if (attr_name == "comm_src") {
        assign_attr_val(node->attr(i), static_cast<any>(&comm_src_));
      } else if (attr_name == "comm_dst") {
        assign_attr_val(node->attr(i), static_cast<any>(&comm_dst_));
      } else if (attr_name == "comm_tag") {
        assign_attr_val(node->attr(i), static_cast<any>(&comm_tag_));
      } else {
        throw runtime_error("unsupported attr");
      }
    } catch (const bad_any_cast& e) {
      std::cerr << "Unmatch attr field type of: node=" << this->id_
                << ", attr_name=" << attr_name << std::endl;
      throw(e);
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

void ETFeederNode::assign_attr_val(
    const ChakraProtoMsg::AttributeProto& attr,
    std::any member) {
  const auto& value_type = attr.value_case();
  if (value_type == ChakraProtoMsg::AttributeProto::kDoubleVal) {
    auto typed_member = any_cast<optional<double>*>(member);
    typed_member->emplace(attr.double_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kDoubleList) {
    auto typed_member = any_cast<optional<vector<double>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.double_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kFloatVal) {
    auto typed_member = any_cast<optional<float>*>(member);
    typed_member->emplace(attr.float_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kFloatList) {
    auto typed_member = any_cast<optional<vector<float>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.float_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kInt32Val) {
    auto typed_member = any_cast<optional<int32_t>*>(member);
    typed_member->emplace(attr.int32_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kInt32List) {
    auto typed_member = any_cast<optional<vector<int32_t>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.int32_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kInt64Val) {
    auto typed_member = any_cast<optional<int64_t>*>(member);
    typed_member->emplace(attr.int64_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kInt64List) {
    auto typed_member = any_cast<optional<vector<int64_t>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.int64_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kUint32Val) {
    auto typed_member = any_cast<optional<uint32_t>*>(member);
    typed_member->emplace(attr.uint32_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kUint32List) {
    auto typed_member = any_cast<optional<vector<uint32_t>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.uint32_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kUint64Val) {
    auto typed_member = any_cast<optional<uint64_t>*>(member);
    typed_member->emplace(attr.uint64_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kUint64List) {
    auto typed_member = any_cast<optional<vector<uint64_t>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.uint64_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kSint32Val) {
    auto typed_member = any_cast<optional<int32_t>*>(member);
    typed_member->emplace(attr.sint32_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kSint32List) {
    auto typed_member = any_cast<optional<vector<int32_t>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.sint32_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kSint64Val) {
    auto typed_member = any_cast<optional<int64_t>*>(member);
    typed_member->emplace(attr.sint64_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kSint64List) {
    auto typed_member = any_cast<optional<vector<int64_t>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.sint64_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kFixed32Val) {
    auto typed_member = any_cast<optional<uint32_t>*>(member);
    typed_member->emplace(attr.fixed32_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kFixed32List) {
    auto typed_member = any_cast<optional<vector<uint32_t>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.fixed32_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kFixed64Val) {
    auto typed_member = any_cast<optional<uint64_t>*>(member);
    typed_member->emplace(attr.fixed64_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kFixed64List) {
    auto typed_member = any_cast<optional<vector<uint64_t>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.fixed64_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kSfixed32Val) {
    auto typed_member = any_cast<optional<int32_t>*>(member);
    typed_member->emplace(attr.sfixed32_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kSfixed32List) {
    auto typed_member = any_cast<optional<vector<int32_t>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.sfixed32_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kSfixed64Val) {
    auto typed_member = any_cast<optional<int64_t>*>(member);
    typed_member->emplace(attr.sfixed64_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kSfixed64List) {
    auto typed_member = any_cast<optional<vector<int64_t>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.sfixed64_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kBoolVal) {
    auto typed_member = any_cast<optional<bool>*>(member);
    typed_member->emplace(attr.bool_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kBoolList) {
    auto typed_member = any_cast<optional<vector<bool>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.bool_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kStringVal) {
    auto typed_member = any_cast<optional<string>*>(member);
    typed_member->emplace(attr.string_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kStringList) {
    auto typed_member = any_cast<optional<vector<string>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.string_list().values()) {
      typed_member->value().push_back(item);
    }
  } else if (value_type == ChakraProtoMsg::AttributeProto::kBytesVal) {
    auto typed_member = any_cast<optional<string>*>(member);
    typed_member->emplace(attr.bytes_val());
  } else if (value_type == ChakraProtoMsg::AttributeProto::kBytesList) {
    auto typed_member = any_cast<optional<vector<string>>*>(member);
    typed_member->emplace();
    for (const auto& item : attr.bytes_list().values()) {
      typed_member->value().push_back(item);
    }
  } else {
    throw runtime_error("undefined attr type in chakra node ");
  }
}

uint64_t ETFeederNode::id() {
  return id_;
}

string ETFeederNode::name() {
  return name_;
}

ChakraProtoMsg::NodeType ETFeederNode::type() {
  return node_->type();
}

bool ETFeederNode::is_cpu_op() {
  if (this->is_cpu_op_.has_value())
    return is_cpu_op_.value();
  throw std::runtime_error(
      "Asked for attr \"is_cpu_op\" from node " + std::to_string(this->id_) +
      ", which do not exists");
}

uint64_t ETFeederNode::runtime() {
  if (this->runtime_.has_value())
    return this->runtime_.value();
  throw std::runtime_error(
      "Asked for attr \"runtime\" from node " + std::to_string(this->id_) +
      ", which do not exists");
}

uint64_t ETFeederNode::num_ops() {
  if (this->num_ops_.has_value())
    return this->num_ops_.value();
  throw std::runtime_error(
      "Asked for attr \"num_ops\" from node " + std::to_string(this->id_) +
      ", which do not exists");
}

uint32_t ETFeederNode::tensor_loc() {
  if (this->tensor_loc_.has_value())
    return this->tensor_loc_.value();
  throw std::runtime_error(
      "Asked for attr \"tensor_loc\" from node " + std::to_string(this->id_) +
      ", which do not exists");
}

uint64_t ETFeederNode::tensor_size() {
  if (this->tensor_size_.has_value())
    return this->tensor_size_.value();
  throw std::runtime_error(
      "Asked for attr \"tensor_size\" from node " + std::to_string(this->id_) +
      ", which do not exists");
}

ChakraProtoMsg::CollectiveCommType ETFeederNode::comm_type() {
  if (this->comm_type_.has_value())
    return this->comm_type_.value();
  throw std::runtime_error(
      "Asked for attr \"comm_type\" from node " + std::to_string(this->id_) +
      ", which do not exists");
}

uint32_t ETFeederNode::involved_dim_size() {
  if (this->involved_dim_size_.has_value())
    return this->involved_dim_size_.value();
  throw std::runtime_error(
      "Asked for attr \"involved_dim_size\" from node " +
      std::to_string(this->id_) + ", which do not exists");
}

bool ETFeederNode::involved_dim(int i) {
  if (this->involved_dim_.has_value())
    if (static_cast<size_t>(i) < this->involved_dim_.value().size())
      return this->involved_dim_.value()[i];
  throw std::runtime_error(
      "Asked for attr \"involved_dim\"[" + std::to_string(i) + "] from node " +
      std::to_string(this->id_) + ", which do not exists");
}

uint32_t ETFeederNode::comm_priority() {
  if (this->comm_priority_.has_value())
    return this->comm_priority_.value();
  throw std::runtime_error(
      "Asked for attr \"comm_priority\" from node " +
      std::to_string(this->id_) + ", which do not exists");
}

uint64_t ETFeederNode::comm_size() {
  if (this->comm_size_.has_value())
    return this->comm_size_.value();
  throw std::runtime_error(
      "Asked for attr \"comm_size\" from node " + std::to_string(this->id_) +
      ", which do not exists");
}

uint32_t ETFeederNode::comm_src() {
  if (this->comm_src_.has_value())
    return this->comm_src_.value();
  throw std::runtime_error(
      "Asked for attr \"comm_src\" from node " + std::to_string(this->id_) +
      ", which do not exists");
}

uint32_t ETFeederNode::comm_dst() {
  if (this->comm_dst_.has_value())
    return this->comm_dst_.value();
  throw std::runtime_error(
      "Asked for attr \"comm_dst\" from node " + std::to_string(this->id_) +
      ", which do not exists");
}

uint32_t ETFeederNode::comm_tag() {
  if (this->comm_tag_.has_value())
    return this->comm_tag_.value();
  throw std::runtime_error(
      "Asked for attr \"comm_tag\" from node " + std::to_string(this->id_) +
      ", which do not exists");
}

bool ETFeederNode::has_is_cpu_op() {
  return this->is_cpu_op_.has_value();
}

bool ETFeederNode::has_runtime() {
  return this->runtime_.has_value();
}

bool ETFeederNode::has_num_ops() {
  return this->num_ops_.has_value();
}

bool ETFeederNode::has_tensor_loc() {
  return this->tensor_loc_.has_value();
}

bool ETFeederNode::has_tensor_size() {
  return this->tensor_size_.has_value();
}

bool ETFeederNode::has_comm_type() {
  return this->comm_type_.has_value();
}

bool ETFeederNode::has_involved_dim_size() {
  return this->involved_dim_size_.has_value();
}

bool ETFeederNode::has_involved_dim(int i) {
  if (!this->has_involved_dim_size())
    return false;
  return static_cast<size_t>(i) < this->involved_dim_.value().size();
}

bool ETFeederNode::has_comm_priority() {
  return this->comm_priority_.has_value();
}

bool ETFeederNode::has_comm_size() {
  return this->comm_size_.has_value();
}

bool ETFeederNode::has_comm_src() {
  return this->comm_src_.has_value();
}

bool ETFeederNode::has_comm_dst() {
  return this->comm_dst_.has_value();
}

bool ETFeederNode::has_comm_tag() {
  return this->comm_tag_.has_value();
}

bool ETFeederNode::is_cpu_op(const bool& default_value) {
  if (this->has_is_cpu_op()) {
    return this->is_cpu_op();
  }
  return default_value;
}

uint64_t ETFeederNode::runtime(const uint64_t& default_value) {
  if (this->has_runtime()) {
    return this->runtime();
  }
  return default_value;
}

uint64_t ETFeederNode::num_ops(const uint64_t& default_value) {
  if (this->has_num_ops()) {
    return this->num_ops();
  }
  return default_value;
}

uint32_t ETFeederNode::tensor_loc(const uint32_t& default_value) {
  if (this->has_tensor_loc()) {
    return this->tensor_loc();
  }
  return default_value;
}

uint64_t ETFeederNode::tensor_size(const uint64_t& default_value) {
  if (this->has_tensor_size()) {
    return this->tensor_size();
  }
  return default_value;
}

ChakraProtoMsg::CollectiveCommType ETFeederNode::comm_type(
    const ChakraProtoMsg::CollectiveCommType& default_value) {
  if (this->has_comm_type()) {
    return this->comm_type();
  }
  return default_value;
}

uint32_t ETFeederNode::involved_dim_size(const uint32_t& default_value) {
  if (this->has_involved_dim_size()) {
    return this->involved_dim_size();
  }
  return default_value;
}

bool ETFeederNode::involved_dim(int i, const bool& default_value) {
  if (this->has_involved_dim(i)) {
    return this->involved_dim(i);
  }
  return default_value;
}

uint32_t ETFeederNode::comm_priority(const uint32_t& default_value) {
  if (this->has_comm_priority()) {
    return this->comm_priority();
  }
  return default_value;
}

uint64_t ETFeederNode::comm_size(const uint64_t& default_value) {
  if (this->has_comm_size()) {
    return this->comm_size();
  }
  return default_value;
}

uint32_t ETFeederNode::comm_src(const uint32_t& default_value) {
  if (this->has_comm_src()) {
    return this->comm_src();
  }
  return default_value;
}

uint32_t ETFeederNode::comm_dst(const uint32_t& default_value) {
  if (this->has_comm_dst()) {
    return this->comm_dst();
  }
  return default_value;
}

uint32_t ETFeederNode::comm_tag(const uint32_t& default_value) {
  if (this->has_comm_tag()) {
    return this->comm_tag();
  }
  return default_value;
}
