// DMA timing only. No weights, tensors, arithmetic results or measured cycles.
// Uses the same pinned Ramulator ABI as transactional_emulator/lib/ramulator.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

extern "C" {
void *ramulator_new(const char *);
bool ramulator_request(void *, uint64_t, bool, void (*)(void *), void *, int);
float ramulator_period(void *);
void ramulator_tick(void *);
}

struct Memory {
    void *ram;
    uint64_t now=0, next_tick=0, pending=0, reads=0, writes=0;
    static void done(void *opaque) { --static_cast<Memory *>(opaque)->pending; }
    void tick() {
        now=std::max(now,next_tick);
        ramulator_tick(ram);
        next_tick=now+1;
    }
    void group(uint64_t base, uint64_t size, bool write) {
        while(next_tick<now) { ramulator_tick(ram); ++next_tick; }
        for(uint64_t offset=0;offset<size;offset+=16) {
            for(;;) {
                ++pending;
                if(ramulator_request(ram,base+offset,write,done,this,16)) break;
                --pending;
                tick();
            }
        }
        while(pending) tick();
        (write?writes:reads)+=size;
    }
};

int main(int argc,char **argv) {
    if(argc!=4) { std::cerr<<"usage: lt_memory config.json trace.txt review|window\n"; return 2; }
    std::ifstream config_file(argv[1]),trace(argv[2]);
    std::string config((std::istreambuf_iterator<char>(config_file)),{});
    if(config.empty()||!trace) return 3;
    Memory memory{ramulator_new(config.c_str())};
    if(!memory.ram || ramulator_period(memory.ram)!=1.0f) return 4;
    bool review=std::string(argv[3])=="review";
    int window=review?1:std::stoi(argv[3]);
    if(window<1||window>64) return 5;
    uint64_t delay=0,addr,size;
    char op;
    while(trace>>op>>addr>>size) {
        if(op=='d') { memory.now+=addr; delay+=addr; continue; }
        if(addr%64||size%64) return 6;
        if(op=='r') memory.group(addr,size,false);
        else if(op=='w') {
            // The DMA splits each Matrix view into 4096-byte Vector rows.
            for(uint64_t row=0;row<size;row+=4096) {
                uint64_t row_size=std::min(uint64_t(4096),size-row);
                for(uint64_t off=0;off<row_size;off+=64*window) {
                    uint64_t n=std::min(uint64_t(64*window),row_size-off);
                    if(review) memory.group(addr+row+off,n,false);
                    memory.group(addr+row+off,n,true);
                }
            }
        } else return 7;
    }
    std::cout<<"{\"dma_cycles\":"<<memory.now-delay
             <<",\"total_cycles\":"<<memory.now
             <<",\"read_bytes\":"<<memory.reads
             <<",\"write_bytes\":"<<memory.writes<<"}\n";
}
