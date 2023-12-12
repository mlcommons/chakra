#pragma once

#include <any>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include "et_def/et_def.pb.h"

namespace Chakra {

class ETFeederNode {
 public:
  ETFeederNode(std::shared_ptr<ChakraProtoMsg::Node> node);
  std::shared_ptr<ChakraProtoMsg::Node> getChakraNode();
  void addChild(std::shared_ptr<ETFeederNode> node);
  std::vector<std::shared_ptr<ETFeederNode>> getChildren();
  void addDepUnresolvedParentID(uint64_t node_id);
  std::vector<uint64_t> getDepUnresolvedParentIDs();
  void setDepUnresolvedParentIDs(
      std::vector<uint64_t> const& dep_unresolved_parent_ids);

  uint64_t id();
  std::string name();
  ChakraProtoMsg::NodeType type();

  bool is_cpu_op();
  uint64_t runtime();
  uint64_t num_ops();
  uint32_t tensor_loc();
  uint64_t tensor_size();
  ChakraProtoMsg::CollectiveCommType comm_type();
  uint32_t involved_dim_size();
  bool involved_dim(int i);
  uint32_t comm_priority();
  uint64_t comm_size();
  uint32_t comm_src();
  uint32_t comm_dst();
  uint32_t comm_tag();

  bool has_is_cpu_op();
  bool has_runtime();
  bool has_num_ops();
  bool has_tensor_loc();
  bool has_tensor_size();
  bool has_comm_type();
  bool has_involved_dim_size();
  bool has_involved_dim(int i);
  bool has_comm_priority();
  bool has_comm_size();
  bool has_comm_src();
  bool has_comm_dst();
  bool has_comm_tag();

  bool is_cpu_op(const bool& default_value);
  uint64_t runtime(const uint64_t& default_value);
  uint64_t num_ops(const uint64_t& default_value);
  uint32_t tensor_loc(const uint32_t& default_value);
  uint64_t tensor_size(const uint64_t& default_value);
  ChakraProtoMsg::CollectiveCommType comm_type(
      const ChakraProtoMsg::CollectiveCommType& default_value);
  uint32_t involved_dim_size(const uint32_t& default_value);
  bool involved_dim(int i, const bool& default_value);
  uint32_t comm_priority(const uint32_t& default_value);
  uint64_t comm_size(const uint64_t& default_value);
  uint32_t comm_src(const uint32_t& default_value);
  uint32_t comm_dst(const uint32_t& default_value);
  uint32_t comm_tag(const uint32_t& default_value);

 private:
  void assign_attr_val(
      const ChakraProtoMsg::AttributeProto& attr,
      std::any member);

  std::shared_ptr<ChakraProtoMsg::Node> node_{nullptr};
  std::unordered_set<std::shared_ptr<ETFeederNode>> children_set_{};
  std::vector<std::shared_ptr<ETFeederNode>> children_vec_{};
  std::vector<uint64_t> dep_unresolved_parent_ids_{};

  // necessary fields
  uint64_t id_;
  std::string name_;

  // optional fields
  std::optional<bool> is_cpu_op_;
  std::optional<uint64_t> runtime_;
  std::optional<uint64_t> num_ops_;
  std::optional<uint32_t> tensor_loc_;
  std::optional<uint64_t> tensor_size_;
  std::optional<ChakraProtoMsg::CollectiveCommType> comm_type_;
  std::optional<uint32_t> involved_dim_size_;
  std::optional<std::vector<bool>> involved_dim_;
  std::optional<uint32_t> comm_priority_;
  std::optional<uint64_t> comm_size_;
  std::optional<uint32_t> comm_src_;
  std::optional<uint32_t> comm_dst_;
  std::optional<uint32_t> comm_tag_;
};

} // namespace Chakra
