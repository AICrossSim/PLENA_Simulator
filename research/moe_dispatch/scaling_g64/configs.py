"""Frozen requested geometries, including explicit independent-lane controls.

No configuration is selected on held-out results. G is the number of PLENA
subarray reduction groups, not the length of a one-cycle vector dot product.
The declared mapping is BLEN=PM and MLEN=PM*G, with column replication.
"""
from .geometry import Core

SCALES = (65536, 131072, 262144)
POLICIES = {
    "static": dict(policy="hard_threshold", bypass=False, prefetch=True, wait_best=False),
    "naive": dict(policy="mdpu", bypass=True, prefetch=True, wait_best=False),
    "no_preconfig": dict(policy="mdpu", bypass=True, prefetch=False, wait_best=True),
    "full": dict(policy="mdpu", bypass=True, prefetch=True, wait_best=True),
}
PROFILES = ("pipelined_tree1", "pipelined_tree2", "serialized_output")


def configurations():
    configs = []
    for scale in SCALES:
        multiple = scale // 65536
        make = lambda m, n: Core.from_shape(m, n, 64)
        def add(role, variant, cores, groups=None, note=""):
            cores = tuple(cores)
            groups = tuple(tuple(g) for g in (groups or [(i,) for i in range(len(cores))]))
            assert sum(c.macs for c in cores) == scale
            if len(groups) == 2:
                assert all(sum(cores[i].macs for i in g) == scale // 2 for g in groups)
            if role == "hetero":
                assert cores[0].pn <= 256
            configs.append(dict(design_id=f"{scale}_{role}_{variant}", scale=scale,
                                role=role, variant=variant, cores=cores, groups=groups,
                                note=note, mapping="BLEN=PM; MLEN=PM*G; column replication",
                                e_pm_le2=(all(cores[i].pm <= 2 for i in groups[1])
                                          if role == "hetero" else None)))
        add("single", "main", [make(4*multiple,256)])
        add("homo", "main", [make(2*multiple,256)]*2)
        for alternative in (False, True):
            p = make(2*multiple,256) if alternative else make(4*multiple,128)
            add("hetero", "alt_p" if alternative else "main", [p,make(multiple,512)],
                note=("Monolithic E; at256K PM=4 violates the proposed narrow-lane PM<=2 rule."
                      if multiple==4 else "Monolithic E."))
            if multiple >= 2:
                e = make(multiple//2,512)
                add("hetero", "alt_p_split_e" if alternative else "split_e", [p,e,e],
                    [(0,),(1,2)],
                    "Two independently executing E lanes, one shared depth8 E FIFO; three physical workers.")
        add("fair_homo_stream", "main", [make(multiple,512)]*2,
            note="Both cores receive the same PN512 stream capability as monolithic E.")
        if multiple>=2:
            e=make(multiple//2,512)
            add("fair_stream_cluster", "split_e", [e]*4, [(0,1),(2,3)],
                "Four independent stream lanes, two shared depth8 FIFOs; same stream lane capability as split E.")
    return configs


def serializable_configs(configs):
    return [{**c, "cores":[core.to_dict() for core in c["cores"]],
             "groups":[list(g) for g in c["groups"]]} for c in configs]
