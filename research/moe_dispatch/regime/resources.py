"""Independent private quotas and packed-transport sensitivity contracts.

W8/W4 are transport/dequantization hypotheses, not validated trained-model
quantization. They use INT codes, group-128 FP16 scales and row32 alignment.
Packed sector coalescing/scale retention are optimistic transport assumptions.
Installed W operands stay BF16: compression does not create extra SRAM slots.
"""
from dataclasses import dataclass, replace
from ..geometry3d.memory import memory_budget as legacy_budget, align, SRAM_BYTES


@dataclass(frozen=True)
class Partition:
    # Fractions apply to canonical core 0; each arena sums to its fixed total.
    w: float = .5
    x: float = .5
    acc: float = .5
    z: float = .5
    wb: float = .5
    xb: float = .5
    ab: float = .5

    def __post_init__(self):
        if any(not 0 < x < 1 for x in self.__dict__.values()):
            raise ValueError("two positive private partitions required")


def split(total, fraction, quantum=1, minimum=1):
    a = max(minimum, min(total-minimum, round(total*fraction/quantum)*quantum))
    return a,total-a


def memory_budget(cores, batch, hidden, max_f, *, partition=None, **kwargs):
    old = legacy_budget(cores,batch,hidden,max_f,**kwargs)
    if partition is None or len(old.cores)==1:
        return old
    dims = tuple(cores)
    if len(dims)!=2:
        raise ValueError("one or two cores supported")
    p=partition
    ws=split(40960,p.w,32,32); xs=split(12288,p.x,32,32)
    ac=split(98304,p.acc,32,32); zs=split(393216,p.z,32,8192)
    wb=split(64,p.wb); xb=split(24,p.xb); ab=split(12,p.ab)
    out=[]
    structures={k:v for k,v in old.structures.items() if not k.startswith('core')}
    limit=kwargs.get('buffer_limit',32)
    for i,(c,m) in enumerate(zip(dims,old.cores)):
        slots=min(limit,ws[i]//c.w_slice_bytes)
        xbuffers=min(2,xs[i]//align(c.x_slice_bytes))
        reason='' if slots and xbuffers else ('W physical tile does not fit' if not slots else 'X physical tile does not fit')
        nm=replace(m,w_slots=slots,w_capacity_bytes=ws[i],x_register_bytes=xs[i],
                   accumulator_bytes=ac[i],z_bytes=zs[i],w_banks=wb[i],x_banks=xb[i],
                   accumulator_banks=ab[i],x_buffers=xbuffers,eligible=not reason,ineligibility_reason=reason)
        out.append(nm)
        structures.update({f'core{i}_W_slots':ws[i],f'core{i}_X_buffers':xs[i],
                           f'core{i}_accumulator_RF':ac[i],f'core{i}_Z_activation':zs[i]})
    total=sum(structures.values())
    global_fit=batch*hidden*2<=524288 and batch*hidden*4<=1048576
    route_fit=batch*kwargs.get('top_k',6)*16+64*64<=16384
    return replace(old,cores=tuple(out),structures=structures,total_bytes=total,
                   slack_bytes=SRAM_BYTES-total,fits=total<=SRAM_BYTES and global_fit and route_fit and all(m.eligible for m in out))


def packed_projection_bytes(n,k,fmt):
    if min(n,k)<=0 or fmt not in ('BF16','W8','W4'):
        raise ValueError('positive tensor and supported format required')
    if fmt=='BF16':
        return n*align(2*k)
    bits=8 if fmt=='W8' else 4
    codes=align((k*bits+7)//8)
    scales=align(((k+127)//128)*2)
    return n*(codes+scales)


def unique_bytes(workload,fmt):
    return sum(2*packed_projection_bytes(e['F'],e['H'],fmt)+
               packed_projection_bytes(e['H'],e['F'],fmt) for e in workload['experts'])
