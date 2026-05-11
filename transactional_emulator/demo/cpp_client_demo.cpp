#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (char c : input) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

int connect_to_server(const std::string& host, const std::string& port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* result = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &result) != 0) {
        throw std::runtime_error("getaddrinfo failed");
    }

    int socket_fd = -1;
    for (addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        socket_fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (socket_fd == -1) {
            continue;
        }
        if (::connect(socket_fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;
        }
        ::close(socket_fd);
        socket_fd = -1;
    }

    freeaddrinfo(result);

    if (socket_fd == -1) {
        throw std::runtime_error("failed to connect to emulator service");
    }
    return socket_fd;
}

void write_all(int socket_fd, const std::string& payload) {
    size_t written = 0;
    while (written < payload.size()) {
        ssize_t rc = ::send(
            socket_fd,
            payload.data() + written,
            payload.size() - written,
            0
        );
        if (rc < 0) {
            throw std::runtime_error("send failed: " + std::string(std::strerror(errno)));
        }
        written += static_cast<size_t>(rc);
    }
}

std::string read_line(int socket_fd) {
    std::string line;
    char ch = 0;
    while (true) {
        ssize_t rc = ::recv(socket_fd, &ch, 1, 0);
        if (rc < 0) {
            throw std::runtime_error("recv failed: " + std::string(std::strerror(errno)));
        }
        if (rc == 0) {
            break;
        }
        if (ch == '\n') {
            break;
        }
        line.push_back(ch);
    }
    return line;
}

std::string send_request(int socket_fd, const std::string& json) {
    write_all(socket_fd, json + "\n");
    return read_line(socket_fd);
}

void print_exchange(int socket_fd, const std::string& label, const std::string& json) {
    std::cout << ">> " << label << "\n" << json << "\n";
    std::cout << "<< " << send_request(socket_fd, json) << "\n\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string host = argc > 1 ? argv[1] : "127.0.0.1";
    const std::string port = argc > 2 ? argv[2] : "7878";
    const std::string opcode_path = argc > 3 ? argv[3] : "";
    const std::string hbm_path = argc > 4 ? argv[4] : "";
    const std::string fpsram_path = argc > 5 ? argv[5] : "";
    const std::string intsram_path = argc > 6 ? argv[6] : "";
    const std::string vram_path = argc > 7 ? argv[7] : "";

    try {
        const int socket_fd = connect_to_server(host, port);

        print_exchange(socket_fd, "ping", R"({"cmd":"ping"})");
        print_exchange(socket_fd, "get_config", R"({"cmd":"get_config"})");

        if (!hbm_path.empty()) {
            print_exchange(
                socket_fd,
                "load_hbm_file",
                "{\"cmd\":\"load_hbm_file\",\"path\":\"" + json_escape(hbm_path) + "\"}"
            );
        }

        if (!fpsram_path.empty()) {
            print_exchange(
                socket_fd,
                "load_fp_sram_file",
                "{\"cmd\":\"load_fp_sram_file\",\"path\":\"" + json_escape(fpsram_path) + "\"}"
            );
        }

        if (!intsram_path.empty()) {
            print_exchange(
                socket_fd,
                "load_int_sram_file",
                "{\"cmd\":\"load_int_sram_file\",\"path\":\"" + json_escape(intsram_path) + "\"}"
            );
        }

        if (!vram_path.empty()) {
            print_exchange(
                socket_fd,
                "load_vram_file",
                "{\"cmd\":\"load_vram_file\",\"path\":\"" + json_escape(vram_path) + "\"}"
            );
        }

        if (!opcode_path.empty()) {
            print_exchange(
                socket_fd,
                "execute_file",
                "{\"cmd\":\"execute_file\",\"path\":\"" + json_escape(opcode_path) + "\"}"
            );
        }

        print_exchange(socket_fd, "get_state", R"({"cmd":"get_state"})");
        print_exchange(socket_fd, "read_vram", R"({"cmd":"read_vram","addr":0})");
        print_exchange(socket_fd, "read_hbm", R"({"cmd":"read_hbm","addr":0,"len":64})");

        ::close(socket_fd);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "cpp_client_demo failed: " << ex.what() << "\n";
        return 1;
    }
}
