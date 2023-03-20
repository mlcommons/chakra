#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "eg_def/eg_def.pb.h"

namespace Chakra {

class EGFeederNode {
  public:
    std::shared_ptr<ChakraProtoMsg::Node> getChakraNode();
    void setChakraNode(std::shared_ptr<ChakraProtoMsg::Node> node);
    void addChild(std::shared_ptr<EGFeederNode> node);
    std::vector<std::shared_ptr<EGFeederNode>> getChildren();
    void addDepUnresolvedParentID(uint64_t node_id);
    std::vector<uint64_t> getDepUnresolvedParentIDs();
    void setDepUnresolvedParentIDs(std::vector<uint64_t> const& dep_unresolved_parent_ids);

  private:
    std::shared_ptr<ChakraProtoMsg::Node> node_{nullptr};
    std::unordered_set<std::shared_ptr<EGFeederNode>> children_set_{};
    std::vector<std::shared_ptr<EGFeederNode>> children_vec_{};
    std::vector<uint64_t> dep_unresolved_parent_ids_{};
};

} // namespace Chakra
