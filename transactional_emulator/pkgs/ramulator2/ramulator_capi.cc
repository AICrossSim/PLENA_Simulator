#include <base/base.h>
#include <base/request.h>
#include <base/config.h>
#include <frontend/frontend.h>
#include <memory_system/memory_system.h>

#include <exception>
#include <iostream>

#include "ramulator_capi.h"

struct ramulator {
    Ramulator::IFrontEnd *frontend;
    Ramulator::IMemorySystem *memory_system;
};

ramulator* ramulator_new(const char *config) {
    try {
        auto val = new ramulator;
        YAML::Node node = YAML::Load(config);
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
    delete val;
}

bool ramulator_request(ramulator *val, uint64_t addr, bool write, void (*callback)(void*), void *data) {
    return val->frontend->receive_external_requests(write, addr, 0, [=](Ramulator::Request &req) {
        callback(data);
    });
}

float ramulator_period(ramulator *val) {
    return val->memory_system->get_tCK();
}

void ramulator_tick(ramulator *val) {
    val->memory_system->tick();
}
