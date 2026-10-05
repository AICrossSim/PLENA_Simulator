"""Technology-independent structural bill. No fabricated mm^2/J estimates."""
from .resources import memory_budget


def features(cores,settings):
    b=memory_budget(cores,128,2048,2816,partition=settings.partition,
        allocation=settings.allocation,buffer_limit=settings.prefetch_slots,profile=settings.fabric)
    return {'main_BF16_multipliers':sum(c.macs for c in cores),
        'FP32_reduction_adders':sum(c.pm*c.pn*(c.pk-1) for c in cores),
        'dot_outputs':sum(c.pm*c.pn for c in cores),'core_controllers':len(cores),
        'private_W_banks':sum(m.w_banks for m in b.cores),
        'private_X_banks':sum(m.x_banks for m in b.cores),
        'private_accumulator_banks':sum(m.accumulator_banks for m in b.cores),
        'installed_storage_bytes':b.total_bytes,
        'register_pipeline_timing_area_available':False,
        'SRAM_macro_area_available':False,'equal_MAC_is_equal_area':False}


def estimate_mm2(cores,settings,technology_model):
    """Require externally synthesized logic AND legal macro lookup tables."""
    if not technology_model or not technology_model.get('synthesis_validated'):
        raise ValueError('synthesis-calibrated logic and SRAM macro model required')
    if not technology_model.get('qualified_geometry_mm2'):
        raise ValueError('per-geometry synthesized datapath/control/routing area required')
    from ..geometry3d.compute import geometry_id
    gid=geometry_id(cores)
    if gid not in technology_model['qualified_geometry_mm2']:
        raise ValueError('geometry missing from technology-qualified synthesis table')
    # Full-chip values must include concrete macro/bank capacities, registers,
    # control, muxes, routing and private replication; a MAC coefficient alone
    # is explicitly insufficient.
    entry=technology_model['qualified_geometry_mm2'][gid]
    required=('datapath','SRAM_macros','control','routing_and_buffer_registers')
    if not all(k in entry and entry[k]>=0 for k in required):
        raise ValueError('incomplete synthesized chip-area ledger')
    return sum(entry[k] for k in required)
