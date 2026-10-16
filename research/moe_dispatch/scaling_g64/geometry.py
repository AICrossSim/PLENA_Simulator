"""Declared replicated PLENA frames for the 64K/128K/256K scaling study.

G is the number of BLEN-wide reduction groups, not the reduction width.
The default shape mapping uses BLEN=PM and MLEN=PM*G.  BLEN=1 is a
custom scalar-subarray extension; these frames carry no PPA certification.
"""
from dataclasses import dataclass

BUDGETS = (65536, 131072, 262144)
BLENS = (1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True, order=True)
class Core:
    b: int
    r: int
    rm: int = 1
    rn: int = 1

    def __post_init__(self):
        if any(type(x) is not int for x in (self.b, self.r, self.rm, self.rn)):
            raise ValueError("Dimensions must be integers")
        if self.b not in BLENS or self.pm not in BLENS:
            raise ValueError("BLEN and PM must be powers of two in [1,64]")
        if self.r < self.b or self.r > 2048 or self.r & (self.r-1) or self.r % self.b:
            raise ValueError("MLEN must be a power of two in [BLEN,2048]")
        if self.rm < 1 or self.rn < 1 or self.pn % 4:
            raise ValueError("Positive replication and PN multiple of four required")
        if self.macs > max(BUDGETS):
            raise ValueError("Core exceeds the largest declared scaling budget")

    @classmethod
    def from_shape(cls, pm, pn, g=64):
        if any(type(x) is not int for x in (pm, pn, g)) or pm <= 0 or pn % pm:
            raise ValueError("Shape requires integer PN divisible by PM")
        return cls(pm, pm*g, 1, pn//pm)

    @property
    def pm(self): return self.b*self.rm
    @property
    def pn(self): return self.b*self.rn
    @property
    def groups(self): return self.r//self.b
    @property
    def macs(self): return self.pm*self.pn*self.groups
    @property
    def shape(self): return self.pm,self.pn,self.groups
    @property
    def native(self): return self.b >= 2 and self.rm == self.rn == 1
    @property
    def blen1_extension(self): return self.b == 1

    def __str__(self):
        return f"{self.pm}×{self.pn}×{self.groups} (BLEN={self.b}, MLEN={self.r})"

    def to_dict(self):
        return {"b":self.b,"r":self.r,"rm":self.rm,"rn":self.rn,
                "pm":self.pm,"pn":self.pn,"groups":self.groups,"macs":self.macs,
                "native":self.native,"blen1_extension":self.blen1_extension}
