// Combinational ingress decision.  Estimates and queue capacity are explicit.
// A smaller finish time wins; the workload affinity resolves exact ties.
module mdpu_selector (
    input logic in_valid,
    input logic prefer_stream,
    input logic bypass_enable,
    input logic wait_best,
    input logic dense_space, stream_space,
    input logic [31:0] dense_drain, stream_drain,
    input logic [31:0] dense_cost, stream_cost,
    output logic accept,
    output logic selected_stream,
    output logic [32:0] dense_finish, stream_finish
);
    always_comb begin
        dense_finish = {1'b0,dense_drain} + {1'b0,dense_cost};
        stream_finish = {1'b0,stream_drain} + {1'b0,stream_cost};
        selected_stream = prefer_stream;
        accept = 1'b0;
        if (bypass_enable) begin
            if (wait_best || (dense_space && stream_space))
                selected_stream = (stream_finish < dense_finish) ||
                    ((stream_finish == dense_finish) && prefer_stream);
            else if (dense_space) selected_stream = 1'b0;
            else if (stream_space) selected_stream = 1'b1;
            accept = in_valid && (wait_best ? (selected_stream ? stream_space : dense_space) : (dense_space || stream_space));
        end else begin
            accept = in_valid && (prefer_stream ? stream_space : dense_space);
        end
    end
endmodule
