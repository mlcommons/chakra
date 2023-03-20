#pragma once

#include <memory>
#include <unordered_map>
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
    std::vector<uint64_t> getDepUnresolvedParentID();
    void setDepUnresolvedParentID(std::vector<uint64_t> dep_unresolved_parent_id);

  private:
    std::shared_ptr<ChakraProtoMsg::Node> node_{nullptr};
    std::vector<std::shared_ptr<EGFeederNode>> children_{};
    std::vector<uint64_t> dep_unresolved_parent_id_{};
};

} // namespace Chakra
