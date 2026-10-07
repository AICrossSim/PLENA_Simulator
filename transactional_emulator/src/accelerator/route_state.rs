//! Four-token Routed-MoE expert-major dispatch state.

use std::collections::BTreeSet;

const BATCH_TOKENS: usize = 4;
const MAX_TOPK: usize = 8;
const RTL_INT_SRAM_DEPTH: usize = 32;
const RTL_FP_SRAM_DEPTH: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Idle,
    Collecting,
    Ready,
    Active,
    Done,
}

#[derive(Clone, Copy, Debug)]
struct Config {
    expert_gp: u8,
    int_base: usize,
    fp_base: usize,
    expert_count: usize,
    topk: usize,
}

#[derive(Clone, Copy, Debug)]
struct RouteEntry {
    expert: u8,
    weight: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct RouteContext {
    expert: u8,
    token_mask: u8,
    weights: [f32; BATCH_TOKENS],
}

pub(super) struct RouteState {
    phase: Phase,
    config: Option<Config>,
    routes: [[Option<RouteEntry>; MAX_TOPK]; BATCH_TOKENS],
    contexts: Vec<RouteContext>,
    next_context: usize,
    current: Option<RouteContext>,
    loop_start_pc: Option<usize>,
}

impl RouteState {
    pub(super) fn new() -> Self {
        Self {
            phase: Phase::Idle,
            config: None,
            routes: [[None; MAX_TOPK]; BATCH_TOKENS],
            contexts: Vec::new(),
            next_context: 0,
            current: None,
            loop_start_pc: None,
        }
    }

    pub(super) fn configure(
        &mut self,
        expert_gp: u8,
        int_base: usize,
        fp_base: usize,
        expert_count: usize,
        topk: usize,
    ) {
        assert!(
            matches!(self.phase, Phase::Idle | Phase::Done),
            "C_ROUTE_BEGIN cannot replace an in-flight route batch"
        );
        assert!(expert_count > 0 && expert_count <= 256);
        assert!(topk > 0 && topk <= MAX_TOPK && topk <= expert_count);
        let entries = BATCH_TOKENS * topk;
        assert!(
            int_base + entries <= RTL_INT_SRAM_DEPTH,
            "C_ROUTE_BEGIN INT SRAM range exceeds RTL depth"
        );
        assert!(
            fp_base + entries <= RTL_FP_SRAM_DEPTH,
            "C_ROUTE_BEGIN FP SRAM range exceeds RTL depth"
        );

        self.phase = Phase::Collecting;
        self.config = Some(Config {
            expert_gp,
            int_base,
            fp_base,
            expert_count,
            topk,
        });
        self.routes = [[None; MAX_TOPK]; BATCH_TOKENS];
        self.contexts.clear();
        self.next_context = 0;
        self.current = None;
        self.loop_start_pc = None;
    }

    pub(super) fn capture_topk(
        &mut self,
        int_base: usize,
        fp_base: usize,
        indices: &[u32],
        weights: &[f32],
    ) {
        if self.phase != Phase::Collecting {
            return;
        }
        let config = self.config.expect("route collection has no configuration");
        assert_eq!(
            indices.len(),
            config.topk,
            "V_TOPK index count mismatches policy"
        );
        assert_eq!(
            weights.len(),
            config.topk,
            "V_TOPK weight count mismatches policy"
        );
        assert!(int_base >= config.int_base && fp_base >= config.fp_base);
        let int_offset = int_base - config.int_base;
        let fp_offset = fp_base - config.fp_base;
        assert_eq!(int_offset, fp_offset, "V_TOPK INT/FP route offsets differ");
        assert_eq!(
            int_offset % config.topk,
            0,
            "V_TOPK route output is not token-aligned"
        );
        let token = int_offset / config.topk;
        assert!(
            token < BATCH_TOKENS,
            "V_TOPK route token is outside the four-token batch"
        );
        assert!(
            self.routes[token][..config.topk]
                .iter()
                .all(Option::is_none),
            "V_TOPK wrote the same route token twice"
        );

        let mut seen = BTreeSet::new();
        for (rank, (&expert, &weight)) in indices.iter().zip(weights).enumerate() {
            assert!(
                (expert as usize) < config.expert_count,
                "V_TOPK expert is outside policy range"
            );
            assert!(
                seen.insert(expert),
                "V_TOPK returned a duplicate expert for one token"
            );
            self.routes[token][rank] = Some(RouteEntry {
                expert: expert as u8,
                weight,
            });
        }

        if self
            .routes
            .iter()
            .all(|row| row[..config.topk].iter().all(Option::is_some))
        {
            self.build_contexts(config.topk);
            self.phase = Phase::Ready;
        }
    }

    fn build_contexts(&mut self, topk: usize) {
        let experts: BTreeSet<u8> = self
            .routes
            .iter()
            .flat_map(|row| row[..topk].iter().flatten().map(|entry| entry.expert))
            .collect();
        assert!(!experts.is_empty(), "route collection contains no experts");

        self.contexts = experts
            .into_iter()
            .map(|expert| {
                let mut context = RouteContext {
                    expert,
                    ..RouteContext::default()
                };
                for token in 0..BATCH_TOKENS {
                    if let Some(entry) = self.routes[token][..topk]
                        .iter()
                        .flatten()
                        .find(|entry| entry.expert == expert)
                    {
                        context.token_mask |= 1 << token;
                        context.weights[token] = entry.weight;
                    }
                }
                context
            })
            .collect();
    }

    pub(super) fn start_loop(&mut self, pc: usize) -> (u8, u8) {
        assert_eq!(
            self.phase,
            Phase::Ready,
            "C_ROUTE_LOOP_START before four valid V_TOPK results"
        );
        assert!(!self.contexts.is_empty());
        self.loop_start_pc = Some(pc);
        self.current = Some(self.contexts[0]);
        self.next_context = 1;
        self.phase = Phase::Active;
        let config = self.config.unwrap();
        (config.expert_gp, self.contexts[0].expert)
    }

    pub(super) fn end_loop(&mut self) -> Option<(usize, u8, u8)> {
        assert_eq!(
            self.phase,
            Phase::Active,
            "C_ROUTE_LOOP_END outside active route loop"
        );
        if self.next_context < self.contexts.len() {
            let context = self.contexts[self.next_context];
            self.next_context += 1;
            self.current = Some(context);
            let config = self.config.unwrap();
            Some((
                self.loop_start_pc.unwrap() + 1,
                config.expert_gp,
                context.expert,
            ))
        } else {
            self.current = None;
            self.phase = Phase::Done;
            None
        }
    }

    pub(super) fn current_route(&self, token: u8) -> (bool, f32) {
        assert!(
            (token as usize) < BATCH_TOKENS,
            "V_ROUTE_MUL token must be in 0..4"
        );
        let context = self.current.expect("V_ROUTE_MUL outside active route loop");
        let active = context.token_mask & (1 << token) != 0;
        (active, context.weights[token as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::RouteState;

    #[test]
    fn groups_four_tokens_in_ascending_expert_order() {
        let mut routes = RouteState::new();
        routes.configure(7, 0, 8, 32, 4);
        let rows = [
            ([7, 2, 9, 12], [0.1, 0.2, 0.3, 0.4]),
            ([2, 5, 7, 20], [0.5, 0.6, 0.7, 0.8]),
            ([7, 3, 18, 5], [0.9, 1.0, 1.1, 1.2]),
            ([12, 20, 5, 9], [1.3, 1.4, 1.5, 1.6]),
        ];
        for (token, (experts, weights)) in rows.iter().enumerate() {
            routes.capture_topk(token * 4, 8 + token * 4, experts, weights);
        }

        assert_eq!(routes.start_loop(20), (7, 2));
        assert_eq!(routes.current_route(0), (true, 0.2));
        assert_eq!(routes.current_route(1), (true, 0.5));
        assert_eq!(routes.current_route(2), (false, 0.0));
        assert_eq!(routes.end_loop(), Some((21, 7, 3)));

        let mut observed = vec![2, 3];
        while let Some((target, gp, expert)) = routes.end_loop() {
            assert_eq!((target, gp), (21, 7));
            observed.push(expert);
        }
        assert_eq!(observed, vec![2, 3, 5, 7, 9, 12, 18, 20]);
    }

    #[test]
    fn supports_qwen_full_capacity() {
        let mut routes = RouteState::new();
        routes.configure(5, 0, 0, 128, 8);
        let experts = [1, 4, 9, 17, 33, 64, 96, 127];
        for token in 0..4 {
            let weights = [token as f32; 8];
            routes.capture_topk(token * 8, token * 8, &experts, &weights);
        }
        assert_eq!(routes.start_loop(3), (5, 1));
        assert_eq!(routes.current_route(3), (true, 3.0));
    }

    #[test]
    fn registered_policy_supports_expert_255() {
        let mut routes = RouteState::new();
        routes.configure(5, 0, 0, 256, 8);
        for token in 0..4 {
            routes.capture_topk(
                token * 8,
                token * 8,
                &[0, 7, 31, 63, 127, 191, 254, 255],
                &[0.125; 8],
            );
        }
        let (_, first) = routes.start_loop(10);
        assert_eq!(first, 0);

        let mut observed = vec![first];
        while let Some((_, _, expert)) = routes.end_loop() {
            observed.push(expert);
        }
        assert_eq!(observed, vec![0, 7, 31, 63, 127, 191, 254, 255]);
    }

    #[test]
    #[should_panic(expected = "duplicate expert")]
    fn rejects_duplicate_expert_within_token() {
        let mut routes = RouteState::new();
        routes.configure(1, 0, 0, 32, 4);
        routes.capture_topk(0, 0, &[3, 3, 4, 5], &[0.25; 4]);
    }

    #[test]
    #[should_panic(expected = "token must be in 0..4")]
    fn rejects_route_mul_token_outside_batch() {
        let mut routes = RouteState::new();
        routes.configure(1, 0, 0, 32, 4);
        for token in 0..4 {
            routes.capture_topk(token * 4, token * 4, &[0, 1, 2, 3], &[0.25; 4]);
        }
        routes.start_loop(0);
        routes.current_route(4);
    }
}
