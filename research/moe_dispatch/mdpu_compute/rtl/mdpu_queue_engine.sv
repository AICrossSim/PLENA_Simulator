// Synthesizable control-path prototype. Cores supply current-task countdowns.
// Queued service scoreboards and shadow context determine remaining drain.
// This block owns both finite FIFOs and shadow contexts, not matrix datapaths.
module mdpu_queue_engine #(
    parameter integer DEPTH=8,
    parameter integer CONTEXT_CYCLES=4,
    parameter integer USE_EXTERNAL_DRAIN=0,
    parameter integer PW=$clog2(DEPTH),
    parameter integer CW=$clog2(DEPTH+1),
    parameter integer SW=$clog2(CONTEXT_CYCLES+1)
) (
    input logic clk, reset,
    input logic bypass_enable, head_bypass_enable, prefetch_enable, wait_best,
    input logic [7:0] threshold,
    input logic in_valid,
    output logic in_ready,
    input logic [6:0] in_expert,
    input logic [7:0] in_tokens,
    input logic in_shared,
    input logic [31:0] in_cost0, in_cost1,
    input logic [31:0] drain0, drain1,
    input logic [31:0] current_remaining0, current_remaining1,
    input logic core_ready0, core_ready1,
    output logic issue_valid0, issue_valid1,
    output logic [6:0] issue_expert0, issue_expert1,
    output logic [31:0] issue_cost0, issue_cost1,
    output logic [CW-1:0] count0, count1,
    output logic shadow_valid0, shadow_valid1,
    output logic accepted_stream,
    output logic stolen0, stolen1,
    output logic claim_valid0, claim_valid1
);
    typedef struct packed {
        logic [6:0] expert;
        logic [7:0] tokens;
        logic shared_task;
        logic [31:0] cost0, cost1;
    } descriptor_t;
    descriptor_t fifo0 [DEPTH], fifo1 [DEPTH];
    descriptor_t shadow0, shadow1, incoming, claimed0, claimed1;
    logic [PW-1:0] rd0,wr0,rd1,wr1;
    logic [SW-1:0] prepare0,prepare1;
    logic accept_internal, prefer_stream;
    logic [32:0] finish0,finish1;
    logic claim0,claim1,pop0,pop1,push0,push1;
    logic source0,source1;
    logic [34:0] own_head_finish0,own_head_finish1;
    logic [34:0] steal_finish0,steal_finish1;
    logic [34:0] base0,base1,post_base0,post_base1,post_current0,post_current1;
    logic [34:0] calc_drain0,calc_drain1;
    logic [31:0] queue_work0,queue_work1,selected_drain0,selected_drain1;
    logic eligible0,eligible1;

    mdpu_selector selector(
        .in_valid(in_valid), .prefer_stream(prefer_stream),
        .wait_best(wait_best),
        .bypass_enable(bypass_enable), .dense_space((count0 - CW'(pop0)) < CW'(DEPTH)),
        .stream_space((count1 - CW'(pop1)) < CW'(DEPTH)), .dense_drain(selected_drain0),
        .stream_drain(selected_drain1), .dense_cost(in_cost0), .stream_cost(in_cost1),
        .accept(accept_internal), .selected_stream(accepted_stream),
        .dense_finish(finish0), .stream_finish(finish1));

    assign incoming = {in_expert,in_tokens,in_shared,in_cost0,in_cost1};
    assign prefer_stream = !in_shared && in_tokens <= threshold;
    assign in_ready = (bypass_enable && !wait_best) ? ((count0-CW'(pop0)) < CW'(DEPTH) || (count1-CW'(pop1)) < CW'(DEPTH)) :
                   (bypass_enable ? (accepted_stream ? (count1-CW'(pop1)) < CW'(DEPTH) : (count0-CW'(pop0)) < CW'(DEPTH)) :
                   (prefer_stream ? (count1-CW'(pop1)) < CW'(DEPTH) : (count0-CW'(pop0)) < CW'(DEPTH)));
    assign push0 = accept_internal && !accepted_stream;
    assign push1 = accept_internal && accepted_stream;
    assign issue_valid0 = shadow_valid0 && prepare0 == 0;
    assign issue_valid1 = shadow_valid1 && prepare1 == 0;
    assign issue_expert0 = shadow0.expert;
    assign issue_expert1 = shadow1.expert;
    assign issue_cost0 = shadow0.cost0;
    assign issue_cost1 = shadow1.cost1;
    always_comb begin
        // Only FIFO heads can move.  CURRENT and NEXT never migrate.
        base0={3'b0,current_remaining0};base1={3'b0,current_remaining1};
        if(shadow_valid0)begin
            if(35'(prepare0)>base0)base0=35'(prepare0);
            base0=base0+{3'b0,shadow0.cost0};
        end
        if(shadow_valid1)begin
            if(35'(prepare1)>base1)base1=35'(prepare1);
            base1=base1+{3'b0,shadow1.cost1};
        end
        own_head_finish0 = (base0>35'(CONTEXT_CYCLES)?base0:35'(CONTEXT_CYCLES)) + {3'b0,fifo0[rd0].cost0};
        own_head_finish1 = (base1>35'(CONTEXT_CYCLES)?base1:35'(CONTEXT_CYCLES)) + {3'b0,fifo1[rd1].cost1};
        post_current0=(issue_valid0&&core_ready0)?{3'b0,shadow0.cost0}:{3'b0,current_remaining0};
        post_current1=(issue_valid1&&core_ready1)?{3'b0,shadow1.cost1}:{3'b0,current_remaining1};
        steal_finish0 = (post_current0 > 35'(CONTEXT_CYCLES) ?
            post_current0 : 35'(CONTEXT_CYCLES)) + {3'b0,fifo1[rd1].cost0};
        steal_finish1 = (post_current1 > 35'(CONTEXT_CYCLES) ?
            post_current1 : 35'(CONTEXT_CYCLES)) + {3'b0,fifo0[rd0].cost1};
        claim0=0; claim1=0; source0=0; source1=1;
        eligible0=(!shadow_valid0 || (issue_valid0 && core_ready0)) &&
            (prefetch_enable || (core_ready0 && !issue_valid0));
        eligible1=(!shadow_valid1 || (issue_valid1 && core_ready1)) &&
            (prefetch_enable || (core_ready1 && !issue_valid1));
        // Own-queue claims have priority. Empty queues cannot grant.
        if (eligible0 && count0 != 0) claim0=1;
        if (eligible1 && count1 != 0) claim1=1;
        if (eligible0 && count0 == 0 && count1 != 0 &&
            !claim1 && head_bypass_enable && steal_finish0 < own_head_finish1) begin
            claim0=1; source0=1;
        end
        if (eligible1 && count1 == 0 && count0 != 0 &&
            !claim0 && head_bypass_enable && steal_finish1 < own_head_finish0) begin
            claim1=1; source1=0;
        end
        claimed0 = source0 ? fifo1[rd1] : fifo0[rd0];
        claimed1 = source1 ? fifo1[rd1] : fifo0[rd0];
        pop0 = (claim0 && !source0) || (claim1 && !source1);
        pop1 = (claim0 && source0) || (claim1 && source1);
        stolen0 = claim0 && source0;
        stolen1 = claim1 && !source1;
        claim_valid0 = claim0; claim_valid1 = claim1;
        // Ingress observes same-clock issue/claim events, as does the runtime.
        post_base0=base0;post_base1=base1;
        if(claim0)post_base0=(post_current0>35'(CONTEXT_CYCLES)?post_current0:35'(CONTEXT_CYCLES))+{3'b0,claimed0.cost0};
        if(claim1)post_base1=(post_current1>35'(CONTEXT_CYCLES)?post_current1:35'(CONTEXT_CYCLES))+{3'b0,claimed1.cost1};
        calc_drain0=post_base0+{3'b0,queue_work0}-(pop0?{3'b0,fifo0[rd0].cost0}:35'd0);
        calc_drain1=post_base1+{3'b0,queue_work1}-(pop1?{3'b0,fifo1[rd1].cost1}:35'd0);
        if(!prefetch_enable)begin
            calc_drain0=calc_drain0+(35'(count0)-35'(pop0))*35'(CONTEXT_CYCLES);
            calc_drain1=calc_drain1+(35'(count1)-35'(pop1))*35'(CONTEXT_CYCLES);
        end
        if(calc_drain0<35'(CONTEXT_CYCLES)+35'd1)calc_drain0=35'(CONTEXT_CYCLES)+35'd1;
        if(calc_drain1<35'(CONTEXT_CYCLES)+35'd1)calc_drain1=35'(CONTEXT_CYCLES)+35'd1;
        selected_drain0=USE_EXTERNAL_DRAIN!=0?drain0:(|calc_drain0[34:32]?32'hffffffff:calc_drain0[31:0]);
        selected_drain1=USE_EXTERNAL_DRAIN!=0?drain1:(|calc_drain1[34:32]?32'hffffffff:calc_drain1[31:0]);
    end
    always_ff @(posedge clk) begin
        if (reset) begin
            rd0<=0;wr0<=0;rd1<=0;wr1<=0;count0<=0;count1<=0;
            shadow_valid0<=0;shadow_valid1<=0;prepare0<=0;prepare1<=0;
            queue_work0<=0;queue_work1<=0;
            shadow0<='0;shadow1<='0;
        end else begin
            queue_work0<=queue_work0+(push0?in_cost0:32'd0)-(pop0?fifo0[rd0].cost0:32'd0);
            queue_work1<=queue_work1+(push1?in_cost1:32'd0)-(pop1?fifo1[rd1].cost1:32'd0);
            if (push0) begin fifo0[wr0]<=incoming;wr0<=wr0+1'b1;end
            if (push1) begin fifo1[wr1]<=incoming;wr1<=wr1+1'b1;end
            if (pop0) rd0<=rd0+1'b1;
            if (pop1) rd1<=rd1+1'b1;
            case ({push0,pop0})
                2'b10:count0<=count0+1'b1;2'b01:count0<=count0-1'b1;
                default: begin end
            endcase
            case ({push1,pop1})
                2'b10:count1<=count1+1'b1;2'b01:count1<=count1-1'b1;
                default: begin end
            endcase
            if (issue_valid0 && core_ready0) shadow_valid0<=0;
            if (issue_valid1 && core_ready1) shadow_valid1<=0;
            if (shadow_valid0 && prepare0 != 0 && (prefetch_enable || core_ready0))
                prepare0<=prepare0-1'b1;
            if (shadow_valid1 && prepare1 != 0 && (prefetch_enable || core_ready1))
                prepare1<=prepare1-1'b1;
            if (claim0) begin
                shadow0<=claimed0;shadow_valid0<=1;
                // First setup word is consumed on the claim edge if enabled.
                prepare0<= (prefetch_enable || core_ready0) ? SW'(CONTEXT_CYCLES-1) : SW'(CONTEXT_CYCLES);
            end
            if (claim1) begin
                shadow1<=claimed1;shadow_valid1<=1;
                prepare1<= (prefetch_enable || core_ready1) ? SW'(CONTEXT_CYCLES-1) : SW'(CONTEXT_CYCLES);
            end
        end
    end
endmodule
