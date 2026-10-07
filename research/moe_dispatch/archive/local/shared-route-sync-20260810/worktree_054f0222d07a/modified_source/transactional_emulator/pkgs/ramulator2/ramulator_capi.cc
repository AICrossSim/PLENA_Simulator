#include <ramulator/base/base.h>
#include <ramulator/base/request.h>
#include <ramulator/base/config.h>
#include <ramulator/frontend/i_frontend.h>
#include <ramulator/memory_system/i_memory_system.h>

#include <exception>
#include <iostream>
#include <string>

#include "ramulator_capi.h"

struct ramulator {
    Ramulator::IFrontEnd *frontend;
    Ramulator::IMemorySystem *memory_system;
};

namespace {

const Ramulator::ConfigNode* find_node_with_id(
    const Ramulator::ConfigNode& node,
    const std::string& id
) {
    if (node.is_map()) {
        auto id_node = node["id"];
        if (id_node && id_node.is_scalar() && id_node.as<std::string>() == id) {
            return &node;
        }
        for (const auto& [_, child] : node.map()) {
            if (const auto* match = find_node_with_id(child, id)) {
                return match;
            }
        }
    } else if (node.is_sequence()) {
        for (const auto& child : node.seq()) {
            if (const auto* match = find_node_with_id(child, id)) {
                return match;
            }
        }
    }
    return nullptr;
}

uint64_t counter(const Ramulator::ConfigNode& node, const char* name) {
    return node[name].as<unsigned long long>(0);
}

}  // namespace

ramulator* ramulator_new(const char *config) {
    try {
        auto val = new ramulator;
        auto node = Ramulator::Config::parse_config_string(config);
        val->frontend = Ramulator::Factory::create_frontend(node);
        val->memory_system = Ramulator::Factory::create_memory_system(node);

        val->frontend->connect_memory_system(val->memory_system);
        val->memory_system->connect_frontend(val->frontend);
        return val;
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return nullptr;
    }
}

void ramulator_finalize(ramulator *val) {
    val->frontend->finalize();
    val->memory_system->finalize();
    delete val->frontend;
    delete val->memory_system;
    delete val;
}

bool ramulator_request(ramulator *val, uint64_t addr, bool write, void (*callback)(void*), void *data, int size) {
    return val->frontend->receive_external_requests(write, addr, 0, [=](Ramulator::Request &req) {
        callback(data);
    }, size);
}

float ramulator_period(ramulator *val) {
    return val->memory_system->get_tCK();
}

void ramulator_tick(ramulator *val) {
    val->memory_system->tick();
}

bool ramulator_get_channel_telemetry(
    ramulator *val,
    uint32_t channel,
    ramulator_channel_telemetry *out
) {
    if (val == nullptr || out == nullptr) {
        return false;
    }
    try {
        const auto tree = val->memory_system->collect_stats();
        const auto* controller = find_node_with_id(
            tree,
            "Channel " + std::to_string(channel)
        );
        if (controller == nullptr) {
            return false;
        }
        *out = ramulator_channel_telemetry{
            counter(*controller, "cycles"),
            counter(*controller, "num_read_reqs"),
            counter(*controller, "num_read_reqs_served"),
            counter(*controller, "num_write_reqs"),
            counter(*controller, "num_write_reqs_served"),
            counter(*controller, "row_hits"),
            counter(*controller, "row_misses"),
            counter(*controller, "row_conflicts"),
        };
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "Ramulator telemetry error: " << ex.what() << std::endl;
        return false;
    }
}
