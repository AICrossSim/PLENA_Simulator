// Exact cost for the declared analytical model, not a physical timing oracle.
// Host/compiler programs coeff = 2*ceil(F/PN)*ceil(H/MLEN)*BLEN
//                              + ceil(H/PN)*ceil(F/MLEN)*BLEN.
// PM is a power of two; tail = two charged phase drains.
// Arbitrary PN requires compiler coefficient generation, not a shift trick.
module mdpu_estimator (
    input logic [7:0] tokens,
    input logic [2:0] pm_log2,
    input logic [31:0] coefficient,
    input logic [15:0] tail,
    output logic [31:0] cycles,
    output logic overflow
);
    logic [15:0] rounded, waves;
    logic [47:0] full_cost;
    always_comb begin
        rounded = {8'b0,tokens} + ((16'd1 << pm_log2) - 16'd1);
        waves = rounded >> pm_log2;
        full_cost = waves * coefficient + {32'b0,tail};
        overflow = |full_cost[47:32];
        cycles = (tokens == 0) ? 32'd0 :
                 (overflow ? 32'hffffffff : full_cost[31:0]);
    end
endmodule
