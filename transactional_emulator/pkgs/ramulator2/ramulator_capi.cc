#include <ramulator/base/base.h>
#include <ramulator/base/request.h>
#include <ramulator/base/config.h>
#include <ramulator/frontend/i_frontend.h>
#include <ramulator/memory_system/i_memory_system.h>

#include <exception>
#include <iostream>
#include <sstream>
#include <cstring>
#include <dlfcn.h>

#include "ramulator_capi.h"

struct ramulator {
    Ramulator::IFrontEnd *frontend;
    Ramulator::IMemorySystem *memory_system;
};

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

uint32_t ramulator_capi_version() { return 2; }

uint32_t ramulator_tx_bytes(ramulator *val) {
    return val->memory_system->get_tx_bytes();
}

uint64_t ramulator_stats(ramulator *val, char *buffer, uint64_t capacity) {
    val->memory_system->update_stats_recursive();
    std::ostringstream stream;
    val->memory_system->print_stats(stream);
    const auto text = stream.str();
    const auto required = text.size() + 1;
    if (buffer && capacity >= required) std::memcpy(buffer, text.c_str(), required);
    return required;
}

const char *ramulator_library_path() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(&ramulator_capi_version), &info)) return nullptr;
    return info.dli_fname;
}
