#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "et_def/et_def.pb.h"

namespace Chakra {

class ETFeederNode {
  public:
    std::shared_ptr<ChakraProtoMsg::Node> getChakraNode();
    void setChakraNode(std::shared_ptr<ChakraProtoMsg::Node> node);
    void addChild(std::shared_ptr<ETFeederNode> node);
    std::vector<std::shared_ptr<ETFeederNode>> getChildren();
    void addDepUnresolvedParentID(uint64_t node_id);
    std::vector<uint64_t> getDepUnresolvedParentIDs();
    void setDepUnresolvedParentIDs(std::vector<uint64_t> const& dep_unresolved_parent_ids);

  private:
    std::shared_ptr<ChakraProtoMsg::Node> node_{nullptr};
    std::unordered_set<std::shared_ptr<ETFeederNode>> children_set_{};
    std::vector<std::shared_ptr<ETFeederNode>> children_vec_{};
    std::vector<uint64_t> dep_unresolved_parent_ids_{};
};

} // namespace Chakra
