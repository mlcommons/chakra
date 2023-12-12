#pragma once

#include <memory>
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
  bool is_cpu_op();
  ChakraProtoMsg::NodeType type();
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

 private:
  void assign_attr_val(
      std::shared_ptr<ChakraProtoMsg::Node> node,
      int i,
      void* member);

  std::shared_ptr<ChakraProtoMsg::Node> node_{nullptr};
  std::unordered_set<std::shared_ptr<ETFeederNode>> children_set_{};
  std::vector<std::shared_ptr<ETFeederNode>> children_vec_{};
  std::vector<uint64_t> dep_unresolved_parent_ids_{};

  uint64_t id_ = 0ul;
  std::string name_ = "";
  bool is_cpu_op_ = false;
  uint64_t runtime_ = 0ul;
  uint64_t num_ops_ = 0ul;
  uint32_t tensor_loc_ = 0u;
  uint64_t tensor_size_ = 0ul;
  ChakraProtoMsg::CollectiveCommType comm_type_ =
      ChakraProtoMsg::CollectiveCommType::BROADCAST;
  uint32_t involved_dim_size_ = 0u;
  std::vector<bool> involved_dim_;
  uint32_t comm_priority_ = 0u;
  uint64_t comm_size_ = 0ul;
  uint32_t comm_src_ = 0u;
  uint32_t comm_dst_ = 0u;
  uint32_t comm_tag_ = 0u;
};

} // namespace Chakra
