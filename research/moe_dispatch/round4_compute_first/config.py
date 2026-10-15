from dataclasses import dataclass
import math
from research.moe_dispatch.round2.model import Parameters as BaseParameters

TOTAL_MACS = 65536
BATCHES = (2, 4, 8, 16, 64, 96, 128)
BANDWIDTHS = (256, 512, 1229, 2458, 9830)
PK_VALUES = (32, 64, 128, 256, 512)

@dataclass(frozen=True, order=True)
class Core:
    pm: int
    pn: int
    pk: int
    def __post_init__(self):
        if not (1 <= self.pm <= 16 and 1 <= self.pn <= 512 and self.pk in PK_VALUES):
            raise ValueError('outside the common round-four shape domain')
    @property
    def macs(self): return self.pm*self.pn*self.pk
    @property
    def record_bytes(self): return 4*self.pm*self.pn
    @property
    def x_slice_bytes(self): return 2*self.pm*self.pk
    @property
    def w_slice_bytes(self): return 2*self.pn*self.pk

@dataclass(frozen=True)
class Parameters(BaseParameters):
    credits: int = 520
    @property
    def hbm_bandwidth(self):
        if math.isinf(self.hbm_Bpc): return float('inf')
        return min(self.hbm_Bpc,self.credits*32/(self.hbm_latency+1))
    def __post_init__(self):
        if self.hbm_latency < 0 or self.hbm_Bpc <= 0 or self.credits <= 0:
            raise ValueError('invalid memory operating point')
        if self.bank_Bpc <= 0 or self.vector_scale <= 0 or self.request_bytes != 32:
            raise ValueError('invalid port settings')

def parameters(bw=None, control=True):
    # B/cycle equals decimal GB/s at 1 GHz. Infinite HBM retains finite SRAM.
    if bw is None:
        return Parameters(hbm_Bpc=float('inf'), credits=10**18, hbm_latency=0,
                          charge_control=control)
    return Parameters(hbm_Bpc=bw, credits=math.ceil(bw*65/32), hbm_latency=64,
                      charge_control=control)
