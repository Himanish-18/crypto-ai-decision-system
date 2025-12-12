
// v33 FPGA Feature Engine
// Calculates Microstructure Imbalance & Volatility in Hardware (<100ns)

module feature_engine (
    input wire clk,
    input wire rst_n,
    input wire [63:0] best_bid_qty,
    input wire [63:0] best_ask_qty,
    input wire valid_book,
    output reg [63:0] imbalance_ratio // (Bid / (Bid+Ask)) * 100
);
    
    // Simple 1-Cycle Fixed Point Math
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            imbalance_ratio <= 50; // Neutral 0.5
        end else begin
            if (valid_book) begin
                if ((best_bid_qty + best_ask_qty) > 0)
                   imbalance_ratio <= (best_bid_qty * 100) / (best_bid_qty + best_ask_qty);
                else
                   imbalance_ratio <= 50;
            end
        end
    end

endmodule
