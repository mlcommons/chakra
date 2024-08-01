#include "et_feeder/JSONNode.h"

// JSONNode default constructor
JSONNode::JSONNode() {}

// JSONNode copy constructor
JSONNode::JSONNode(const JSONNode &t) {
	node_id = t.node_id;
	node_name = t.node_name;
	node_type = t.node_type;
	is_cpu_op = t.is_cpu_op;
	runtime = t.runtime;
	data_deps = t.data_deps;	
	dep_unresolved_parent_ids_json = t.dep_unresolved_parent_ids_json;
	children_vec_json = t.children_vec_json;
	children_set_json = t.children_set_json;
	if (node_type == 5 || node_type == 6 || node_type == 7) {
		tensor_size = t.tensor_size;
		comm_type = t.comm_type;
		comm_priority = t.comm_priority;
		comm_size = t.comm_size;
		comm_src = t.comm_src;
		comm_dst = t.comm_dst;
		comm_tag = t.comm_tag;
	}
}

// JSONNode constructor
JSONNode::JSONNode(json data, int32_t id) {
			try {
				node_id = data["workload_graph"][id]["Id"];
			}
			catch (...) {
				std::cerr << "node_id not specified in ET" << std::endl;
			}
			try {
				node_name = data["workload_graph"][id]["Name"];
			}
			catch (...) {
				std::cerr << "node_name not specified in ET" << std::endl;
			}
			try {
				node_type = data["workload_graph"][id]["NodeType"];
			}
			catch (...) {
				std::cerr << "node_type not specified in ET" << std::endl;
			}
			try {
				is_cpu_op = data["workload_graph"][id]["is_cpu_op"];
			}
			catch (...) {
				std::cerr << "is_cpu_op not specified in ET" << std::endl;
			}
			try {
				runtime = data["workload_graph"][id]["runtime"];
			}
			catch (...) {}
			try {
				data_deps = data["workload_graph"][id]["data_deps"].get<std::vector<int64_t>>();
			}
			catch (...) {
				std::cerr << "data deps not specified in ET" << std::endl;
			}
			if (node_type == 5 || node_type == 6 || node_type == 7) {
				try {
					tensor_size = data["workload_graph"][id]["tensor_size"];
				}
				catch (...) {}
				try {
					comm_type = data["workload_graph"][id]["comm_type"];
				}
				catch (...) {}
				try {
					comm_priority = data["workload_graph"][id]["comm_priority"];
				}
				catch (...) {}
				try {
					comm_size = data["workload_graph"][id]["comm_size"];
				}
				catch (...) {}
				try {
					comm_src = data["workload_graph"][id]["comm_src"];
				}
				catch (...) {}
				try {
					comm_dst = data["workload_graph"][id]["comm_dst"];
				}
				catch (...) {}
				try {
					comm_tag = data["workload_graph"][id]["comm_tag"];
				}
				catch (...) {}
			}
		}

// Dependency unresolved parent IDs
void JSONNode::addDepUnresolvedParentID(int64_t node_id) {
	dep_unresolved_parent_ids_json.emplace_back(node_id);
}

// Get dependency unresolved parent IDs
std::vector<int64_t> JSONNode::getDepUnresolvedParentIDs() {
	return dep_unresolved_parent_ids_json;
}

// Set dependency unresolved parent IDs
void JSONNode::setDepUnresolvedParentIDs(std::vector<int64_t> const& dep_unresolved_parent_ids) {
	dep_unresolved_parent_ids_json = dep_unresolved_parent_ids;
}

// Add child
void JSONNode::addChild(JSONNode node) {
	 // Avoid adding the same child node multiple times
	// addChild is called multiple times to resolve dependencies
	if (children_set_json.find(node) != children_set_json.end()) {
		return;
	}
	children_vec_json.emplace_back(node);
	children_set_json.emplace(node);
}

// Get children vector
std::vector<JSONNode> JSONNode::getChildren() {
  return children_vec_json;
}
