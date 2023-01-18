#pragma once

#include <memory>
#include <vector>

#include "eg_def/eg_def.pb.h"

namespace Chakra {

class EGFeederNode {
  public:
    std::shared_ptr<ChakraProtoMsg::Node> getChakraNode();
    void setChakraNode(std::shared_ptr<ChakraProtoMsg::Node> node);
    void addChild(std::shared_ptr<EGFeederNode> node);
    std::vector<std::shared_ptr<EGFeederNode>> getChildren();

  private:
    std::shared_ptr<ChakraProtoMsg::Node> node_{nullptr};
    std::vector<std::shared_ptr<EGFeederNode>> children_{};
};

} // namespace Chakra
