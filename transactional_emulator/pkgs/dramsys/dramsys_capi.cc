#include <unordered_map>
#include <utility>
#include <deque>

#include <systemc>
#include <tlm>
#include <tlm_utils/peq_with_cb_and_phase.h>
#include <tlm_utils/simple_initiator_socket.h>

#include <DRAMSys/config/DRAMSysConfiguration.h>
#include <DRAMSys/simulation/DRAMSys.h>
#include <DRAMSys/common/MemoryManager.h>
#include "dramsys_capi.h"

struct dramsys: sc_core::sc_module {
    DRAMSys::DRAMSys sys;
    DRAMSys::MemoryManager memoryManager;
    tlm_utils::peq_with_cb_and_phase<dramsys> eventQueue;
    tlm_utils::simple_initiator_socket<dramsys> iSocket;
    std::deque<tlm::tlm_generic_payload *> sendQueue;

    // Used to synchronize requests to ensure new ones are only issued when acknowledged.
    bool requestInProgress = false;

    std::unordered_map<tlm::tlm_generic_payload *, std::pair<void (*)(void*), void *>> map;

    dramsys(
        const sc_core::sc_module_name& name,
        const DRAMSys::Config::Configuration& config
    ):
        sc_module(name),
        sys("DRAMSys", config),
        memoryManager(false),
        eventQueue(this, &dramsys::queueCallback)
    {
        iSocket.register_nb_transport_bw(this, &dramsys::nb_transport_bw);
        iSocket.bind(sys.tSocket);
    }

    tlm::tlm_sync_enum nb_transport_bw(tlm::tlm_generic_payload& payload,
                                       tlm::tlm_phase& phase,
                                       sc_core::sc_time& bwDelay)
    {
        eventQueue.notify(payload, phase, bwDelay);
        return tlm::TLM_ACCEPTED;
    }


    void queueCallback(tlm::tlm_generic_payload& payload, const tlm::tlm_phase& phase) {
        if (phase == tlm::BEGIN_REQ) {
            // If a request is in progress, queue it up.
            if (requestInProgress) {
                sendQueue.push_back(&payload);
                return;
            }

            tlm::tlm_phase phase = tlm::BEGIN_REQ;
            auto delay = sc_core::SC_ZERO_TIME;
            iSocket->nb_transport_fw(payload, phase, delay);
            requestInProgress = true;
        }

        if (phase == tlm::END_REQ) {
            if (sendQueue.empty()) {
                requestInProgress = false;
                return;
            }

            tlm::tlm_generic_payload* payload = sendQueue.front();
            tlm::tlm_phase phase = tlm::BEGIN_REQ;
            auto delay = sc_core::SC_ZERO_TIME;
            iSocket->nb_transport_fw(*payload, phase, delay);
            sendQueue.pop_front();
        }

        if (phase == tlm::BEGIN_RESP) {
            // Invoke callback.
            auto node = map.extract(&payload);
            node.mapped().first(node.mapped().second);

            // Send END_RESP
            tlm::tlm_phase phase = tlm::END_RESP;
            sc_core::sc_time tCK = sys.getMemSpec().tCK;
            iSocket->nb_transport_fw(payload, phase, tCK);
            payload.release();
        }
    }
};

dramsys* dramsys_new(const char *config_path) {
    auto path = std::filesystem::path { config_path };
    auto config = DRAMSys::Config::from_path(path);

    return new dramsys("dramsys_capi", config);
}

void dramsys_finalize(dramsys *val) {
    delete val;
}

bool dramsys_request(dramsys *val, uint64_t addr, bool write, void (*callback)(void*), void *data) {
    // Allocate payload and initialize.
    tlm::tlm_generic_payload* payload = val->memoryManager.allocate(64);
    payload->acquire();
    payload->set_address(addr);
    payload->set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
    payload->set_dmi_allowed(false);
    payload->set_byte_enable_length(0);
    payload->set_data_length(64);
    payload->set_streaming_width(64);
    payload->set_command(write ? tlm::TLM_WRITE_COMMAND : tlm::TLM_READ_COMMAND);

    // Fill in the mapping.
    val->map[payload] = { callback, data };

    // Queue it up for sending.
    tlm::tlm_phase phase = tlm::BEGIN_REQ;
    auto delay = sc_core::SC_ZERO_TIME;
    val->eventQueue.notify(*payload, phase, delay);
    return true;
}

float dramsys_period(dramsys *val) {
    sc_core::sc_time tCK = val->sys.getMemSpec().tCK;
    return tCK.to_seconds() * 1e9;
}

void dramsys_tick(dramsys *val) {
    sc_core::sc_time tCK = val->sys.getMemSpec().tCK;
    sc_core::sc_start(tCK);
}
