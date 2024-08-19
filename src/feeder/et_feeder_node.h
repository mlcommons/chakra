#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "et_def.pb.h"

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

  const ChakraProtoMsg::AttributeProto& get_other_attr(
      const std::string& attr_name) const;
  bool has_other_attr(const std::string& attr_name) const;

  uint64_t id() const;
  std::string name() const;
  bool is_cpu_op() const;
  ChakraProtoMsg::NodeType type() const;
  uint64_t runtime() const;
  uint64_t num_ops() const;
  uint32_t tensor_loc() const;
  uint64_t tensor_size() const;
  ChakraProtoMsg::CollectiveCommType comm_type() const;
  uint32_t comm_priority() const;
  uint64_t comm_size() const;
  uint32_t comm_src() const;
  uint32_t comm_dst() const;
  uint32_t comm_tag() const;

 private:
  std::shared_ptr<ChakraProtoMsg::Node> node_{nullptr};
  std::unordered_set<std::shared_ptr<ETFeederNode>> children_set_{};
  std::vector<std::shared_ptr<ETFeederNode>> children_vec_{};
  std::vector<uint64_t> dep_unresolved_parent_ids_{};
  std::unordered_map<std::string, const ChakraProtoMsg::AttributeProto&>
      other_attrs_{};

  uint64_t id_;
  std::string name_;
  bool is_cpu_op_;
  uint64_t runtime_;
  uint64_t num_ops_;
  uint32_t tensor_loc_;
  uint64_t tensor_size_;
  ChakraProtoMsg::CollectiveCommType comm_type_;
  uint32_t comm_priority_;
  uint64_t comm_size_;
  uint32_t comm_src_;
  uint32_t comm_dst_;
  uint32_t comm_tag_;
};

} // namespace Chakra
