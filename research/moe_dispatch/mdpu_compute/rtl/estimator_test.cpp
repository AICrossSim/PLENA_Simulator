#include "Vmdpu_estimator.h"
#include "verilated.h"
#include <cstdint>
#include <iostream>
#include <stdexcept>
int main() {
    Vmdpu_estimator d; unsigned n=0;
    const uint32_t coefficients[]={1,4,27,123,1024,11999,65536,0xffffffffu};
    for(unsigned m=0;m<256;++m) for(unsigned shift=0;shift<7;++shift)
    for(auto coeff:coefficients) for(unsigned tail:{0u,38u,65535u}) {
        d.tokens=m;d.pm_log2=shift;d.coefficient=coeff;d.tail=tail;d.eval();
        uint64_t full=uint64_t((m+(1u<<shift)-1)>>shift)*coeff+tail;
        uint32_t result=m==0?0:(full>0xffffffffULL?0xffffffffu:uint32_t(full));
        if(d.cycles!=result || d.overflow!=(full>0xffffffffULL))
            throw std::runtime_error("estimator mismatch");
        ++n;
    }
    std::cout << "{\"estimator_vectors\":"<<n<<",\"passed\":true}\n";
}
