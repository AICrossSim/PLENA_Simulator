#include "Vmdpu_selector.h"
#include "verilated.h"
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
int main() {
    Vmdpu_selector d; std::mt19937 rng(65536); unsigned n=0;
    for (int i=0;i<200000;++i) {
        d.in_valid=rng()&1;d.prefer_stream=rng()&1;d.bypass_enable=rng()&1;
        d.wait_best=rng()&1;
        d.dense_space=rng()&1;d.stream_space=rng()&1;
        d.dense_drain=rng();d.stream_drain=rng();d.dense_cost=rng();d.stream_cost=rng();
        if (i%7==0) {d.stream_drain=d.dense_drain;d.stream_cost=d.dense_cost;}
        d.eval();
        uint64_t a=uint64_t(d.dense_drain)+d.dense_cost;
        uint64_t b=uint64_t(d.stream_drain)+d.stream_cost;
        bool chosen=d.prefer_stream,accept=false;
        if (d.bypass_enable) {
            if(d.wait_best || (d.dense_space && d.stream_space)) chosen=b<a || (a==b && d.prefer_stream);
            else if(d.dense_space) chosen=false;
            else if(d.stream_space) chosen=true;
            accept=d.in_valid && (d.wait_best ? (chosen?d.stream_space:d.dense_space) : (d.dense_space || d.stream_space));
        } else accept=d.in_valid && (chosen?d.stream_space:d.dense_space);
        if(d.dense_finish!=a || d.stream_finish!=b || d.selected_stream!=chosen || d.accept!=accept)
            throw std::runtime_error("selector mismatch including 33-bit overflow/ties");
        ++n;
    }
    std::cout << "{\"selector_vectors\":"<<n<<",\"passed\":true}\n";
}
