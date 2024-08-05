#pragma once

#include "json.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <functional>
#include <set>

using json = nlohmann::json;

class JSONNode {
	private:
		int64_t node_id;
		std::string node_name;
		int node_type;
		bool is_cpu_op;
		int64_t runtime;
		int64_t num_ops;
		int64_t tensor_size;
		int64_t comm_type;
		int32_t comm_priority;
		int64_t comm_size;
		int32_t comm_src;
		int32_t comm_dst;
		int32_t comm_tag;
		int32_t involved_dim_size;
		std::vector<bool> involved_dim;

	public:
		std::vector<int64_t> data_deps{};
		std::vector<int64_t> dep_unresolved_parent_ids_json{};
		std::vector<JSONNode> children_vec_json{};

		// Compare function for set
		struct CompareJSONNodesLT {
			bool operator()(const JSONNode& a, const JSONNode& b) const {
				return a.node_id < b.node_id;
			}
		};
		std::set<JSONNode, CompareJSONNodesLT> children_set_json{};
		
		JSONNode();
		JSONNode(const JSONNode &t);
		JSONNode(json data, int32_t id);
		int64_t id() const;
		std::string name() const;
		int type() const;
		bool isCPUOp() const;
		int64_t getRuntime() const;
		int64_t getNumOps() const;
		int64_t getTensorSize() const;
		int64_t getCommType() const;
		int32_t getCommPriority() const;
		int64_t getCommSize() const;
		int32_t getCommSrc() const;
		int32_t getCommDst() const;
		int32_t getCommTag() const;
		int32_t getInvolvedDimSize() const;
		bool getInvolvedDim(int i) const;
		void addDepUnresolvedParentID(int64_t node_id);
		std::vector<int64_t> getDepUnresolvedParentIDs();
		void setDepUnresolvedParentIDs(std::vector<int64_t> const& dep_unresolved_parent_ids);
		void addChild(JSONNode node);
		std::vector<JSONNode> getChildren();

		// Define the == operator for comparison 
		bool operator==(const JSONNode& other) const {
			return node_id == other.node_id &&
				node_name == other.node_name &&
				node_type == other.node_type &&
				is_cpu_op == other.is_cpu_op &&
				runtime == other.runtime &&
				num_ops == other.num_ops &&
				tensor_size == other.tensor_size &&
				comm_type == other.comm_type &&
				comm_priority == other.comm_priority &&
				comm_size == other.comm_size &&
				comm_src == other.comm_src &&
				comm_dst == other.comm_dst &&
				comm_tag == other.comm_tag &&
				involved_dim_size == other.involved_dim_size &&
				involved_dim == other.involved_dim &&
				data_deps == other.data_deps &&
				dep_unresolved_parent_ids_json == other.dep_unresolved_parent_ids_json &&
				children_vec_json == other.children_vec_json &&
				children_set_json == other.children_set_json;
		}

		// Overload the assignment operator
		JSONNode& operator=(const JSONNode& other) {
			if (this != &other) {
				// Copy all member variables
				node_id = other.node_id;
				node_name = other.node_name;
				node_type = other.node_type;
				is_cpu_op = other.is_cpu_op;
				runtime = other.runtime;
				num_ops = other.num_ops;
				tensor_size = other.tensor_size;
				comm_type = other.comm_type;
				comm_priority = other.comm_priority;
				comm_size = other.comm_size;
				comm_src = other.comm_src;
				comm_dst = other.comm_dst;
				comm_tag = other.comm_tag;
				involved_dim_size = other.involved_dim_size;
				involved_dim = other.involved_dim;
				data_deps = other.data_deps;
				dep_unresolved_parent_ids_json = other.dep_unresolved_parent_ids_json;
				children_vec_json = other.children_vec_json;
				children_set_json = other.children_set_json;
			}
			return *this;
		}
};

// Define a custom hash function for unordered set
namespace std {
	template<>
	struct hash<JSONNode> {
		std::size_t operator()(const JSONNode& node) const {
			std::size_t h1 = std::hash<int64_t>()(node.id());
			std::size_t h2 = std::hash<std::string>()(node.name());
			std::size_t h3 = std::hash<int>()(node.type());
			std::size_t h4 = std::hash<bool>()(node.isCPUOp());
			std::size_t h5 = std::hash<int64_t>()(node.getRuntime());

			// A prime number for bit manipulation
			const std::size_t prime = 31;

			// Combine the hash of the current member with the hashes of the previous members
			std::size_t hash = h1;
			hash = hash * prime + h2;
			hash = hash * prime + h3;
			hash = hash * prime + h4;
			hash = hash * prime + h5;

			return hash;
		}
	};
}

// Compare function for JSON node for priority queue
struct CompareJSONNodesGT : public std::binary_function<
                          JSONNode,
                          JSONNode,
                          bool> {
  bool operator()(
      const JSONNode lhs,
      const JSONNode rhs) const {
    return lhs.id() > rhs.id();
  }
};