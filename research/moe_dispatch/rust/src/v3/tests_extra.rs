//! Causal integration checks, including cases absent from the large shape study.
use super::*;
fn offload_fragmentation_input(routed:Vec<Value>)->Value {
    let mut experts=vec![json!({"id":-1,"Me":96,"H":2048,"F":2816,"is_shared":true})];experts.extend(routed);
    json!({"workload":{"id":"helper-context-fragmentation","batch":96,"hidden":2048,"top_k":1,"experts":experts},"config":{"lanes":[4,2],"dataflow":["ws_group","is_stream"],"precision":"P1","comp_mode":"offload","rank_lanes":8,"ranks":{"shared":[32,32,48],"routed":[32,32,24]},"pool_bytes":65536,"t_chunk":96,"z_mode":"auto","placement":"fifo","control_costs":false}})
}
fn fragmented_shared_owner(e:&mut Engine)->usize {
    let z=e.z_arena.alloc(2816).unwrap();let u=e.u_arena.alloc(272).unwrap();
    e.bind();let owner=e.cores[0].cur.unwrap();assert!(e.activate_context(owner));
    e.z_arena.release(z,2816);e.u_arena.release(u,272);
    e.tasks[owner].action=e.tasks[owner].plan.actions.iter().position(|a|matches!(a,Action::Group(g) if g.kind==Kind::Main)).unwrap();
    e.cores[0].next=Some(owner);e.control_free=0;owner
}
#[test]
fn fragmented_helper_next_waits_through_completion_and_run_promotion() {
    let q=offload_fragmentation_input(vec![json!({"id":7,"Me":1,"H":2048,"F":1408}),json!({"id":8,"Me":4,"H":2048,"F":1408})]);
    let mut e=Engine::new(&q).unwrap();let owner=fragmented_shared_owner(&mut e);
    e.bind();let helper=e.cores[1].cur.unwrap();e.tasks[helper].predicted=1;e.control_free=0;e.bind();
    let next=e.cores[1].next.unwrap();assert!(e.tasks[next].z_addr.is_none());
    e.complete(1,helper);
    assert!(e.cores[1].cur.is_none());assert_eq!(e.cores[1].next,Some(next));assert!(e.tasks[next].start.is_none());
    let before=(e.z_arena.ranges.clone(),e.u_arena.ranges.clone(),e.cores[1].acc_arena.ranges.clone(),e.last_progress);
    assert!(!e.promote_next(1));assert_eq!(before,(e.z_arena.ranges.clone(),e.u_arena.ranges.clone(),e.cores[1].acc_arena.ranges.clone(),e.last_progress));
    // This is the gate used by run(): failed promotion leaves the helper
    // available for Shared, whose actual retirement then unblocks Next.
    e.tasks[owner].action=e.tasks[owner].plan.actions.iter().position(|a|matches!(a,Action::Group(g) if g.kind==Kind::Tail)).unwrap();
    assert_eq!(e.helper_owner(),Some(0));assert!(e.ensure_helper_acc(owner,1));
    let (_,a,bytes)=e.tasks[owner].helper_acc.take().unwrap();e.cores[1].acc_arena.release(a,bytes);
    e.cores[0].next=None;e.complete(0,owner);assert!(e.promote_next(1));
    assert_eq!(e.cores[1].cur,Some(next));assert!(e.cores[1].next.is_none());assert_eq!(e.tasks[next].z_addr,Some(0));assert_eq!(e.tasks[next].u_addr,Some(0));assert_eq!(e.tasks[next].acc_addr,Some(0));
}
#[test]
fn fragmented_current_eligibility_keeps_other_window_candidates_available() {
    for policy in ["fifo","supply_ipd"] {
        let q=offload_fragmentation_input(vec![json!({"id":7,"Me":4,"H":2048,"F":1408}),json!({"id":8,"Me":1,"H":2048,"F":1408})]);
        let mut e=Engine::new(&q).unwrap();fragmented_shared_owner(&mut e);
        e.cfg.placement=policy.into();e.bind();
        let t=e.cores[1].cur.unwrap();assert_eq!(e.experts[e.tasks[t].expert].id,8);assert_eq!(e.pending.iter().copied().collect::<Vec<_>>(),vec![1]);
        assert_eq!(e.tasks[t].z_addr,Some(0));assert_eq!(e.tasks[t].u_addr,Some(0));
    }
}
#[test]
fn joint_revalidates_contiguous_current_eligibility_without_changing_anchor_policy() {
    let q=offload_fragmentation_input(vec![json!({"id":7,"Me":4,"H":2048,"F":1408}),json!({"id":8,"Me":1,"H":2048,"F":1408})]);
    let mut e=Engine::new(&q).unwrap();fragmented_shared_owner(&mut e);e.cfg.placement="joint".into();e.cfg.trace=true;
    e.bind();let proposal=e.trace.iter().find(|v|v["event"]=="joint_proposal").unwrap();
    assert_eq!(proposal["eligible_masks"],json!([0,2]));assert!(e.cores[1].cur.is_none());
    // A previously proposed idle-helper placement is cancelled if its
    // physical reservation no longer fits. Busy preferred-core proposals
    // retain the existing Joint anchor/wait policy.
    e.joint_pending.clear();e.joint_pending.push_back((1,1));e.control_free=0;e.bind();
    assert!(e.joint_pending.is_empty());assert_eq!(e.joint_cancellations,1);assert!(e.cores[1].cur.is_none());
}
#[test]
fn candidate_context_metadata_matches_exact_plans() {
    for precision in ["P0","P1","P2"] {for mode in ["none","lanes","separate","kext","offload"] {for zmode in ["auto","full","streamed"] {for flow in ["ws_group","is_stream","switchable"] {
        let q=json!({"workload":{"batch":96,"hidden":544,"top_k":1,"experts":[{"id":-1,"Me":96,"H":544,"F":1280,"is_shared":true},{"id":7,"Me":4,"H":544,"F":640,"is_shared":false}]},"config":{"lanes":[4,2],"dataflow":["ws_group",flow],"precision":precision,"comp_mode":mode,"factor_a":"mxint4","factor_b":"mxint4","rank_lanes":8,"ranks":{"shared":[16,24,32],"routed":[8,16,24]},"t_chunk":96,"z_mode":zmode,"comp_equal_bytes":true,"inline_silu":false}});
        let Ok(e)=Engine::new(&q)else{continue;};
        for task in 0..e.experts.len(){for c in 0..e.cores.len(){if !e.legal(task,c){continue;}
            if zmode=="auto"{let budget=e.storage()["structures"]["Z_dense"].as_u64().unwrap()as usize;assert_eq!(e.binding_z_mode(task),if e.experts[task].m*e.experts[task].f*2>budget{"streamed"}else{"full"});}
            let Some((_,p))=e.binding_plan(task,c)else{continue;};let (z,u)=e.plan_context_footprint(task,&p);
            assert_eq!(e.binding_context_footprint(task,c),(e.plan_accumulator_footprint(task,&p),z,u));
        }}
    }}}}
    let e=Engine::new(&offload_fragmentation_input(vec![json!({"id":7,"Me":4,"H":2048,"F":1408})])).unwrap();
    for task in 0..e.experts.len(){for c in 0..e.cores.len(){if e.legal(task,c){let (_,p)=e.binding_plan(task,c).unwrap();let(z,u)=e.plan_context_footprint(task,&p);assert_eq!(e.binding_context_footprint(task,c),(e.plan_accumulator_footprint(task,&p),z,u));}}}
    for chunk in [32,64,96,128]{let mut q=offload_fragmentation_input(vec![json!({"id":7,"Me":4,"H":2048,"F":1408})]);q["config"]["t_chunk"]=json!(chunk);let Ok(e)=Engine::new(&q)else{continue;};let budget=e.storage()["structures"]["Z_dense"].as_u64().unwrap()as usize;for task in 0..e.experts.len(){assert_eq!(e.binding_z_mode(task),if e.experts[task].m*e.experts[task].f*2>budget{"streamed"}else{"full"});}}
    let values=json!([100.,50.,0.,0.,0.]);let costs=json!([0.,10.,20.,30.,40.]);
    for selection in ["common","independent_projection"] {for precision in ["P1","P2"] {
        let q=json!({"workload":{"batch":1,"hidden":544,"top_k":1,"experts":[{"id":0,"Me":1,"H":544,"F":32,"route_scores":[1.]}]},"config":{"lanes":[6],"precision":precision,"comp_mode":"lanes","rank_lanes":8,"rank_alloc":"gate_weighted","rank_selection":selection,"z_mode":"auto","lambda0":1.,"rank_energy_table":{"0":{"gate":values,"up":values,"down":values},"factor_bytes_per_projection":{"gate":[costs],"up":[costs],"down":[costs]}}}});
        let e=Engine::new(&q).unwrap();let (_,p)=e.binding_plan(0,0).unwrap();assert!(p.ranks.iter().any(|&r|r>0));let(z,u)=e.plan_context_footprint(0,&p);assert_eq!(e.binding_context_footprint(0,0),(e.plan_accumulator_footprint(0,&p),z,u));
    }}
}
#[test]
fn failed_steal_preserves_source_context_and_accepted_response_leases() {
    let q=json!({"workload":{"id":"steal-context-proof","batch":8,"hidden":544,"top_k":1,"experts":[{"id":0,"Me":8,"H":544,"F":160},{"id":1,"Me":2,"H":544,"F":32}]},"config":{"lanes":[4,2],"dataflow":["ws_group","ws_group"],"precision":"P0","comp_mode":"none","control_costs":false,"placement":"fifo","contexts_per_core":2,"context_interleave":false,"max_cycles":100000}});
    let mut e=Engine::new(&q).unwrap();e.bind();e.tasks[0].predicted=1;
    e.cores[1].cur=Some(0);e.cores[1].next=Some(0);e.control_free=0;e.bind();e.cores[1].cur=None;e.cores[1].next=None;e.tasks[0].predicted=1000000;
    let next=e.cores[0].next.unwrap();let old=e.tasks[next].offset;e.cores[0].quota=e.cfg.pool;
    for _ in 0..1000 {let mut busy=vec![false;e.cfg.pool_banks];e.returns();e.land(&mut busy);e.admit();e.hbm();e.now+=1;if e.tiles[old].sent>0{break;}}
    assert!(e.tiles[old].sent>e.tiles[old].landed);assert!(e.tasks[next].acc_addr.is_some());
    let capacity=e.cores[1].acc_arena.capacity;let bytes=e.accumulator_footprint(next);let half=bytes/2;
    e.cores[1].acc_arena.ranges=vec![(0,half),(capacity-half,half)];e.cores[1].acc_arena.used=capacity-bytes;
    let before=(e.tasks[next].acc_addr,e.tasks[next].z_addr,e.tasks[next].u_addr,e.z_arena.ranges.clone(),e.u_arena.ranges.clone(),e.tiles[old].addr,e.tiles[old].sent,e.control_free,e.last_progress);
    e.work_steal();assert_eq!(e.steals,0);assert_eq!(e.cores[0].next,Some(next));assert!(e.cores[1].cur.is_none());assert_eq!(e.tasks[next].offset,old);assert!(!e.tiles[old].discarded);
    assert_eq!(before,(e.tasks[next].acc_addr,e.tasks[next].z_addr,e.tasks[next].u_addr,e.z_arena.ranges.clone(),e.u_arena.ranges.clone(),e.tiles[old].addr,e.tiles[old].sent,e.control_free,e.last_progress));
    // The occupying actor retires. The same steal now reserves a real
    // destination context and drains its original accepted DMA normally.
    e.cores[1].acc_arena=Pool::new(capacity,1);e.work_steal();assert_eq!(e.steals,1);assert!(e.tasks[next].acc_addr.is_some());assert!(e.tasks[next].z_addr.is_some());
    let r=e.run().unwrap();assert_eq!(r["drained"],true);assert_eq!(r["dma_transactions_accepted"],r["dma_transactions_landed"]);
}
#[test]
fn alternate_context_waits_for_real_contiguous_reservations() {
    let q=json!({"workload":{"id":"alternate-context-proof","batch":4,"hidden":512,"top_k":1,"experts":[{"id":-1,"Me":4,"H":512,"F":16,"is_shared":true},{"id":7,"Me":1,"H":512,"F":16}]},"config":{"lanes":[6],"dataflow":["ws_group"],"precision":"P0","comp_mode":"none","rank_lanes":0,"placement":"fifo","contexts_per_core":2,"context_aging":0,"control_costs":false}});
    let mut e=Engine::new(&q).unwrap();e.bind();let cur=e.cores[0].cur.unwrap();e.tasks[cur].predicted=1;e.control_free=0;e.bind();let next=e.cores[0].next.unwrap();
    let g=e.group_at(next,0).unwrap();for i in g.first_tile..g.first_tile+g.tile_count{e.tiles[e.tasks[next].offset+i].ready=true;}
    let (cz,nz)=(e.context_footprint(cur).0,e.context_footprint(next).0);
    // Recreate valid Current ownership with the prospective Next footprint
    // split around it, rather than replacing an arena under a live address.
    let old=e.tasks[cur].z_addr.take().unwrap();e.z_arena.release(old,cz);
    e.z_arena=Pool::new(cz+nz,1);let prefix=e.z_arena.alloc(nz/2).unwrap();let ca=e.z_arena.alloc(cz).unwrap();e.z_arena.release(prefix,nz/2);e.tasks[cur].z_addr=Some(ca);
    e.now=100;e.alternate_context(cur);assert_eq!(e.cores[0].cur,Some(cur));assert_eq!(e.cores[0].next,Some(next));assert!(e.tasks[next].z_addr.is_none());assert_eq!(e.context_switches,0);
}
#[test]
fn fragmented_context_cannot_install_a_helper_current() {
    let q=json!({"workload":{"id":"helper-context-fragmentation","batch":96,"hidden":2048,"top_k":1,"experts":[{"id":-1,"Me":96,"H":2048,"F":2816,"is_shared":true},{"id":7,"Me":4,"H":2048,"F":1408,"is_shared":false}]},"config":{"lanes":[4,2],"dataflow":["ws_group","is_stream"],"precision":"P1","comp_mode":"offload","rank_lanes":8,"ranks":{"shared":[32,32,48],"routed":[32,32,24]},"pool_bytes":65536,"t_chunk":96,"z_mode":"auto","placement":"fifo","control_costs":false}});
    let mut e=Engine::new(&q).unwrap();
    assert_eq!(e.z_arena.capacity,207872);assert_eq!(e.u_arena.capacity,41024);
    // A preceding Me1 context occupied the prefix when Shared activated.
    // Its real retirement leaves enough total bytes, split into two holes.
    let prefix_z=e.z_arena.alloc(2816).unwrap();let prefix_u=e.u_arena.alloc(272).unwrap();
    e.bind();let owner=e.cores[0].cur.unwrap();assert!(e.activate_context(owner));
    e.z_arena.release(prefix_z,2816);e.u_arena.release(prefix_u,272);
    assert_eq!(e.z_arena.ranges,vec![(0,2816),(199424,8448)]);
    assert_eq!(e.u_arena.ranges,vec![(0,272),(40208,816)]);
    e.tasks[owner].action=e.tasks[owner].plan.actions.iter().position(|a|matches!(a,Action::Group(g) if g.kind==Kind::Main)).unwrap();
    // Keep the owner's Next occupied so this bind targets the idle helper.
    e.cores[0].next=Some(owner);e.control_free=0;
    let before=(e.z_arena.ranges.clone(),e.u_arena.ranges.clone(),e.cores[1].acc_arena.ranges.clone(),e.tiles.len());
    e.bind();
    assert!(e.cores[1].cur.is_none(),"an unactivatable Current pins Shared offload progress");
    assert_eq!(e.pending.iter().copied().collect::<Vec<_>>(),vec![1]);assert_eq!(e.tasks.len(),1);
    assert_eq!(before,(e.z_arena.ranges.clone(),e.u_arena.ranges.clone(),e.cores[1].acc_arena.ranges.clone(),e.tiles.len()));
    assert_eq!(e.cores[1].acc_arena.used,0);assert_eq!(e.cores[1].wor_live,0);
    // Once the producer really releases its context, the same descriptor
    // becomes Current with all contiguous reservations already installed.
    e.cores[0].next=None;e.complete(0,owner);e.control_free=0;e.bind();
    let helper=e.cores.iter().find_map(|c|c.cur).unwrap();assert_eq!(e.context_footprint(helper),(11264,1088));
    assert_eq!(e.tasks[helper].z_addr,Some(0));assert_eq!(e.tasks[helper].u_addr,Some(0));assert_eq!(e.tasks[helper].acc_addr,Some(0));
}
#[test]
fn fragmented_full_frame_admission_has_no_partial_side_effects() {
    let q=json!({"workload":{"id":"physical-fragmentation","batch":8,"hidden":512,"top_k":1,"experts":[{"id":-1,"Me":8,"H":512,"F":16,"is_shared":true}]},"config":{"lanes":[6],"dataflow":["ws_group"],"wor_tiles":[8],"precision":"P0","comp_mode":"none","rank_lanes":0,"w_reuse":false,"pool_bytes":65536,"placement":"fifo"}});
    let mut e=Engine::new(&q).unwrap();e.bind();e.cores[0].quota=65536;
    assert_eq!(e.group_at(0,0).unwrap().tile_count,8);
    // Exactly 32 KiB free, but only seven full 4-KiB placements: the last
    // 4096 B are split into 3072 and 1024 B, as in the captured failure.
    e.pool.ranges=vec![(0,28672),(32768,3072),(64512,1024)];e.pool.used=32768;
    let before=e.pool.ranges.clone();e.admit();
    assert_eq!(e.tasks[0].admit,0);assert_eq!(e.pool.used,32768);
    assert_eq!(e.pool.ranges,before);assert_eq!(e.tasks[0].bytes_live,0);
    assert!(e.tiles.iter().all(|x|x.addr.is_none()&&x.sent==0));
}
#[test]
fn shared_byte_pool_blocked_head_can_resume_a_ready_parked_frame() {
    let q=json!({"workload":{"id":"ready-parked","batch":4,"hidden":512,"top_k":1,"experts":[{"id":-1,"Me":4,"H":512,"F":16,"is_shared":true},{"id":7,"Me":1,"H":512,"F":16,"is_shared":false}]},"config":{"lanes":[6],"dataflow":["ws_group"],"wor_tiles":[8],"precision":"P0","comp_mode":"none","rank_lanes":0,"pool_bytes":65536,"placement":"fifo","contexts_per_core":2,"context_aging":0,"control_costs":false}});
    let mut e=Engine::new(&q).unwrap();e.bind();let cur=e.cores[0].cur.unwrap();e.tasks[cur].predicted=1;e.control_free=0;e.bind();let parked=e.cores[0].next.unwrap();
    let g=e.group_at(parked,0).unwrap();for i in g.first_tile..g.first_tile+g.tile_count {e.tiles[e.tasks[parked].offset+i].ready=true;}
    e.now=100;e.tasks[cur].blocked=0;e.step_core(0);
    assert_eq!(e.cores[0].cur,Some(parked));assert_eq!(e.cores[0].next,Some(cur));
    assert_eq!(e.context_switches,1);assert!(e.tasks[parked].z_addr.is_some());
}
#[test]
fn offload_frame_cannot_pin_a_busy_helpers_progress() {
    let q=json!({"workload":{"id":"helper-head-protection","batch":8,"hidden":544,"top_k":1,"experts":[{"id":-1,"Me":8,"H":544,"F":160,"is_shared":true},{"id":7,"Me":1,"H":544,"F":160,"is_shared":false}]},"config":{"lanes":[4,2],"dataflow":["ws_group","ws_group"],"wor_tiles":[8,8],"precision":"P1","comp_mode":"offload","rank_lanes":8,"ranks":{"shared":[8,8,8],"routed":[8,8,8]},"pool_bytes":65536,"placement":"fifo","control_costs":false}});
    let mut e=Engine::new(&q).unwrap();e.bind();let owner=e.cores[0].cur.unwrap();
    // Bind the helper during the owner's local Main phase, then model the
    // transition to its next helper-serviced group while that task is busy.
    e.tasks[owner].action=e.tasks[owner].plan.actions.iter().position(|a|matches!(a,Action::Group(g) if g.kind==Kind::Main)).unwrap();
    e.cores[0].next=Some(owner);e.control_free=0;e.bind();e.cores[0].next=None;
    let helper=e.cores[1].cur.unwrap();e.tasks[owner].action=0;
    assert_eq!(e.group_at(owner,0).unwrap().kind,Kind::Prepass);
    e.admit();assert_eq!(e.tasks[owner].admit,0);
    assert!(e.tasks[helper].admit>0);
    assert!(e.tiles[e.tasks[owner].offset..e.tasks[owner].offset+e.tasks[owner].plan.tiles.len()].iter().all(|t|t.addr.is_none()));
}
#[test]
fn prefetched_next_reserves_current_and_next_activation_arenas_together() {
    let q=json!({"workload":{"id":"activation-safe-state","batch":4,"hidden":512,"top_k":1,"experts":[{"id":-1,"Me":4,"H":512,"F":16,"is_shared":true},{"id":7,"Me":1,"H":512,"F":16,"is_shared":false}]},"config":{"lanes":[6],"dataflow":["ws_group"],"wor_tiles":[8],"precision":"P0","comp_mode":"none","rank_lanes":0,"pool_bytes":65536,"placement":"fifo","contexts_per_core":2,"control_costs":false}});
    let mut e=Engine::new(&q).unwrap();e.bind();let cur=e.cores[0].cur.unwrap();e.tasks[cur].predicted=1;e.control_free=0;e.bind();let next=e.cores[0].next.unwrap();
    let (cz,_)=e.context_footprint(cur);let (nz,_)=e.context_footprint(next);
    let ca=e.tasks[cur].z_addr.take().unwrap();e.z_arena.release(ca,cz);
    let aa=e.tasks[cur].acc_addr.take().unwrap();let ab=e.accumulator_footprint(cur);e.cores[0].acc_arena.release(aa,ab);e.tasks[cur].u_addr=None;
    e.z_arena=Pool::new(cz+nz-16,1);e.cores[0].quota=65536;
    e.admit();assert_eq!(e.tasks[next].admit,0);assert!(e.tasks[next].z_addr.is_none());
    assert!(e.activate_context(cur));assert!(!e.activate_context(next));
}
#[test]
fn parked_offload_next_preserves_the_future_owners_helper_accumulator() {
    let q=json!({"workload":{"id":"helper-future-arena","batch":96,"hidden":2048,"top_k":1,"experts":[{"id":-1,"Me":96,"H":2048,"F":2816,"is_shared":true},{"id":7,"Me":1,"H":2048,"F":1408,"is_shared":false},{"id":8,"Me":3,"H":2048,"F":1408,"is_shared":false}]},"config":{"lanes":[4,2],"dataflow":["ws_group","is_stream"],"precision":"P1","comp_mode":"offload","rank_lanes":8,"ranks":{"shared":[32,32,48],"routed":[32,32,24]},"pool_bytes":65536,"t_chunk":96,"z_mode":"streamed","placement":"fifo","control_costs":false}});
    let mut e=Engine::new(&q).unwrap();e.bind();let owner=e.cores[0].cur.unwrap();
    e.tasks[owner].action=e.tasks[owner].plan.actions.iter().position(|a|matches!(a,Action::Group(g) if g.kind==Kind::Main)).unwrap();
    e.cores[0].next=Some(owner);e.control_free=0;e.bind();let helper=e.cores[1].cur.unwrap();e.tasks[helper].predicted=1;e.control_free=0;e.bind();e.cores[0].next=None;e.tasks[owner].action=0;
    let next=e.cores[1].next.unwrap();assert_eq!(e.accumulator_footprint(helper),11264);assert_eq!(e.accumulator_footprint(next),33792);assert_eq!(e.core_acc_capacity(1),45056);
    for core in &mut e.cores {core.quota=65536;}
    e.admit();assert_eq!(e.tasks[next].admit,0);assert!(e.tasks[next].acc_addr.is_none());
    assert!(e.activate_context(helper));
    assert!(e.cores[1].acc_arena.alloc(96*32*4).is_some());
}
#[test]
fn refill_admission_keeps_the_entire_current_group_progress_space() {
    let q=json!({"workload":{"id":"retained-refill-group","batch":8,"hidden":512,"top_k":1,"experts":[{"id":-1,"Me":8,"H":512,"F":16,"is_shared":true},{"id":7,"Me":1,"H":512,"F":16,"is_shared":false}]},"config":{"lanes":[4,2],"dataflow":["ws_group","ws_group"],"wor_tiles":[8,8],"precision":"P0","comp_mode":"none","rank_lanes":0,"w_reuse":false,"pipeline_supply":true,"prefetch_quota":true,"pool_bytes":65536,"placement":"fifo"}});
    let mut e=Engine::new(&q).unwrap();e.bind();e.control_free=0;e.bind();
    let current=e.cores[0].cur.unwrap();
    let next=e.cores[1].cur.take().or(e.cores[0].next).unwrap();
    e.cores[0].next=Some(next);e.tasks[next].core=0;e.cores[0].quota=65536;
    let group=e.group_at(current,0).unwrap();assert_eq!(group.tile_count,8);
    assert_eq!(e.group_at(next,0).unwrap().tile_count,8);
    // Seven live retained weights and another consumer's 4-KiB lease leave
    // 32 KiB free. An eight-tile Next would fit alone, but pin the missing
    // Current tile. Its reservation must wait as a complete group.
    e.pool.alloc(4096).unwrap();
    for i in 0..7 {
        let tile=e.tasks[current].offset+group.first_tile+i;
        let bytes=e.tiles[tile].spec.bytes;assert_eq!(bytes,4096);
        e.tiles[tile].addr=Some(e.pool.alloc(bytes).unwrap());
        e.tiles[tile].reserved_bytes=bytes;e.tiles[tile].ready=true;
        e.tasks[current].admit+=1;e.tasks[current].bytes_live+=bytes;e.cores[0].live+=bytes;
    }
    e.admit();
    assert_eq!(e.tasks[current].admit,8);
    assert_eq!(e.tasks[next].admit,0);
    assert_eq!(e.pool.used,9*4096);
    assert!(e.tiles[e.tasks[current].offset+7].addr.is_some());
}
fn input() -> Value {
    json!({"workload":{"id":"causal","batch":10,"hidden":544,"top_k":1,"experts":[{"id":1,"Me":2,"H":544,"F":32,"is_shared":false,"token_indices":[0,9]}]},"config":{"lanes":[6],"dataflow":["ws_group"],"precision":"P0","comp_mode":"none","rank_lanes":0,"record_trace":true,"trace_limit":20000,"t_chunk":96,"dot_latency":20}})
}
#[test]
fn nonadjacent_routed_rows_use_actual_gather_strides() {
    let r = run(&input()).unwrap();
    let loads: Vec<_> = r["trace"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v["event"] == "x_load" && v["source"] == 0 && v["k_segment"] == 0)
        .collect();
    assert!(!loads.is_empty());
    let spans = loads[0]["spans"].as_array().unwrap();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0][0], 0);
    assert_eq!(spans[1][0], 9 * 544 * 2);
    assert_eq!(spans[0][1], 512 * 2);
    assert_eq!(spans[1][1], 512 * 2);
    assert_eq!(r["invariants"]["gather_addresses_validated"], true);
}
#[test]
fn frozen_physical_capacity_does_not_shrink_with_batch() {
    let a = run(&input()).unwrap();
    let mut q = input();
    q["workload"]["batch"] = json!(16);
    let b = run(&q).unwrap();
    assert_eq!(a["budget"]["total_bytes"], b["budget"]["total_bytes"]);
    assert_eq!(a["budget"]["structures"], b["budget"]["structures"]);
    assert_eq!(a["budget"]["t_chunk_capacity"], 96);
}
#[test]
fn physical_mac_latency_remains_with_ideal_supply() {
    let mut a = input();
    a["config"]["ideal_hbm"] = json!(true);
    a["config"]["ideal_onchip"] = json!(true);
    a["config"]["dot_latency"] = json!(1);
    let fast = run(&a).unwrap();
    a["config"]["dot_latency"] = json!(20);
    let real = run(&a).unwrap();
    assert!(real["cycles"].as_u64().unwrap() > fast["cycles"].as_u64().unwrap());
    let issues: Vec<_> = real["trace"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v["event"] == "issue")
        .collect();
    assert!(!issues.is_empty());
    for e in issues {
        assert!(e["accumulator_complete"].as_u64().unwrap() >= e["cycle"].as_u64().unwrap() + 20);
    }
}
#[test]
fn bounded_wor_and_ordered_feedback_all_complete() {
    let r = run(&input()).unwrap();
    assert_eq!(r["invariants"]["k_order_preserved"], true);
    assert!(
        r["actual_storage_peak_bytes"].as_u64().unwrap()
            <= r["budget"]["capacity_bytes"].as_u64().unwrap()
    );
    for core in r["cores"].as_array().unwrap() {
        assert!(
            core["wor_peak_slots"].as_u64().unwrap()
                <= core["wor_capacity_slots"].as_u64().unwrap()
        );
        assert_eq!(core["state_sum"], r["cycles"]);
    }
    assert_eq!(r["dma_transactions_accepted"], r["dma_transactions_landed"]);
}
#[test]
fn equal_payload_compensation_is_real_dma_not_posthoc_time() {
    let mut q = input();
    q["config"]["precision"] = json!("P1");
    q["config"]["main_bits"] = json!(4);
    q["config"]["rank_lanes"] = json!(8);
    q["config"]["ranks"] = json!({"routed":[8,8,8],"shared":[8,8,8]});
    q["config"]["comp_equal_bytes"] = json!(true);
    q["config"]["comp_mode"] = json!("none");
    let none = run(&q).unwrap();
    q["config"]["comp_mode"] = json!("lanes");
    let lane = run(&q).unwrap();
    assert_eq!(none["weight_bytes"], lane["weight_bytes"]);
    assert_eq!(
        none["dma_transactions_accepted"],
        lane["dma_transactions_accepted"]
    );
    assert!(none["comp_padding_bytes"].as_u64().unwrap() > 0);
    assert_eq!(none["pool_write_bytes"], none["weight_bytes"]);
}
#[test]
fn mxint8_split_prepass_counts_both_passes_and_drains() {
    let mut q = input();
    q["config"]["precision"] = json!("P2");
    q["config"]["factor_a"] = json!("mxint8s");
    q["config"]["factor_b"] = json!("bf16");
    q["config"]["rank_lanes"] = json!(8);
    q["config"]["comp_mode"] = json!("lanes");
    q["config"]["ranks"] = json!({"routed":[8,8,8],"shared":[8,8,8]});
    q["config"]["max_cycles"] = json!(100000);
    let two = run(&q).unwrap();
    q["config"]["factor_a"] = json!("mxint4");
    let one = run(&q).unwrap();
    let n = |r: &Value| {
        r["cores"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["prepass_issues"].as_u64().unwrap())
            .sum::<u64>()
    };
    assert_eq!(n(&two), 2 * n(&one));
    assert_eq!(two["drained"], true);
}
