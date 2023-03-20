#pragma once

#include <memory>
#include <queue>
#include <unordered_map>

#include "third_party/utils/protoio.hh"
#include "eg_feeder/eg_feeder_node.h"

namespace Chakra {

class EGFeeder {
 public:
  EGFeeder(std::string filename);
  ~EGFeeder();

  void addNode(std::shared_ptr<EGFeederNode> node);
  void removeNode(uint64_t node_id);
  bool hasNodesToIssue();
  std::shared_ptr<EGFeederNode> getNextIssuableNode();
  void pushBackIssuableNode(uint64_t node_id);
  std::shared_ptr<EGFeederNode> lookupNode(uint64_t node_id);
  void freeChildrenNodes(uint64_t node_id);

 private:
  std::shared_ptr<EGFeederNode> readNode();
  void readNextWindow();
  void resolveDep();

  ProtoInputStream trace_;
  const uint32_t window_size_;
  bool eg_complete_;

  std::unordered_map<uint64_t, std::shared_ptr<EGFeederNode>> dep_graph_{};
  std::set<uint64_t> dep_free_node_id_set_{};
  std::queue<std::shared_ptr<EGFeederNode>> dep_free_node_queue_{};
  std::set<std::shared_ptr<EGFeederNode>> dep_unresolved_set{};
};

} // namespace Chakra
