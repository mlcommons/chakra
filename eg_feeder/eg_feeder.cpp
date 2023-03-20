#include "eg_feeder/eg_feeder.h"

using namespace std;
using namespace Chakra;

EGFeeder::EGFeeder(string filename)
  : trace_(filename), window_size_(4096), eg_complete_(false) {
  readNextWindow();
}

EGFeeder::~EGFeeder() {
}

void EGFeeder::addNode(shared_ptr<EGFeederNode> node) {
  dep_graph_[node->getChakraNode()->id()] = node;
}

void EGFeeder::removeNode(uint64_t node_id) {
  dep_graph_.erase(node_id);

  if (!eg_complete_
      && (dep_free_node_queue_.size() < window_size_)) {
    readNextWindow();
  }
}

bool EGFeeder::hasNodesToIssue() {
  return !(dep_graph_.empty() && dep_free_node_queue_.empty());
}

shared_ptr<EGFeederNode> EGFeeder::getNextIssuableNode() {
  if (dep_free_node_queue_.size() != 0) {
    shared_ptr<EGFeederNode> node = dep_free_node_queue_.front();
    dep_free_node_id_set_.erase(node->getChakraNode()->id());
    dep_free_node_queue_.pop();
    return node;
  } else {
    return nullptr;
  }
}

void EGFeeder::pushBackIssuableNode(uint64_t node_id) {
  shared_ptr<EGFeederNode> node = dep_graph_[node_id];
  dep_free_node_id_set_.emplace(node_id);
  dep_free_node_queue_.emplace(node);
}

shared_ptr<EGFeederNode> EGFeeder::lookupNode(uint64_t node_id) {
  return dep_graph_[node_id];
}

void EGFeeder::freeChildrenNodes(uint64_t node_id) {
  shared_ptr<EGFeederNode> node = dep_graph_[node_id];
  for (auto child: node->getChildren()) {
    auto child_chakra = child->getChakraNode();
    for (auto it = child_chakra->mutable_parent()->begin();
        it != child_chakra->mutable_parent()->end();
        ++it) {
      if (*it == node_id) {
        child_chakra->mutable_parent()->erase(it);
        break;
      }
    }
    if (child_chakra->parent().size() == 0) {
      dep_free_node_id_set_.emplace(child_chakra->id());
      dep_free_node_queue_.emplace(child);
    }
  }
}

shared_ptr<EGFeederNode> EGFeeder::readNode() {
  shared_ptr<EGFeederNode> node = make_shared<EGFeederNode>();
  shared_ptr<ChakraProtoMsg::Node> pkt_msg = make_shared<ChakraProtoMsg::Node>();

  if (!trace_.read(*pkt_msg)) {
    return nullptr;
  }
  node->setChakraNode(pkt_msg);

  bool dep_unresolved = false;
  for (int i = 0; i < pkt_msg->parent_size(); ++i) {
    auto parent_node = dep_graph_.find(pkt_msg->parent(i));
    if (parent_node != dep_graph_.end()) {
      parent_node->second->addChild(node);
    } else {
      dep_unresolved = true;
      node->addDepUnresolvedParentID(pkt_msg->parent(i));
    }
  }

  if (dep_unresolved) {
    dep_unresolved_node_set_.emplace(node);
  }

  return node;
}

void EGFeeder::resolveDep() {
  for (auto it = dep_unresolved_node_set_.begin();
      it != dep_unresolved_node_set_.end();) {
    shared_ptr<EGFeederNode> node = *it;
    vector<uint64_t> dep_unresolved_parent_ids = node->getDepUnresolvedParentIDs();
    for (auto inner_it = dep_unresolved_parent_ids.begin();
        inner_it != dep_unresolved_parent_ids.end();) {
      auto parent_node = dep_graph_.find(*inner_it);
      if (parent_node != dep_graph_.end()) {
        parent_node->second->addChild(node);
        inner_it = dep_unresolved_parent_ids.erase(inner_it);
      } else {
        ++inner_it;
      }
    }
    if (dep_unresolved_parent_ids.size() == 0) {
      it = dep_unresolved_node_set_.erase(it);
    } else {
      node->setDepUnresolvedParentIDs(dep_unresolved_parent_ids);
      ++it;
    }
  }
}

void EGFeeder::readNextWindow() {
  uint32_t num_read = 0;
  do {
    shared_ptr<EGFeederNode> new_node = readNode();
    if (new_node == nullptr) {
      eg_complete_ = true;
      break;
    }

    addNode(new_node);
    ++num_read;

    resolveDep();
  } while ((num_read < window_size_)
      || (dep_unresolved_node_set_.size() != 0));

  for (auto node_id_node: dep_graph_) {
    uint64_t node_id = node_id_node.first;
    shared_ptr<EGFeederNode> node = node_id_node.second;
    if ((dep_free_node_id_set_.count(node_id) == 0)
        && (node->getChakraNode()->parent().size() == 0)) {
      dep_free_node_id_set_.emplace(node_id);
      dep_free_node_queue_.emplace(node);
    }
  }
}
