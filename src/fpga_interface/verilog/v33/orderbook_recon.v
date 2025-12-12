
// v33 FPGA Order Book Reconstruction
// Decodes UDP Tick -> Updates Bids/Asks SRAM -> Triggers Strategy

module orderbook_recon (
    input wire clk,
    input wire rst_n,
    input wire [63:0] tick_price,
    input wire [63:0] tick_qty,
    input wire [0:0]  tick_side, // 0=Buy, 1=Sell
    input wire valid_in,
    output reg [63:0] best_bid,
    output reg [63:0] best_ask,
    output reg valid_out
);

    // SRAM Simulation (Registers for Top of Book)
    reg [63:0] bid_cache;
    reg [63:0] ask_cache;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            best_bid <= 0;
            best_ask <= 1000000; // Max price
            valid_out <= 0;
        end else begin
            if (valid_in) begin
                if (tick_side == 0) begin
                    // Update Bid
                    if (tick_price > best_bid) best_bid <= tick_price;
                end else begin
                    // Update Ask
                    if (tick_price < best_ask) best_ask <= tick_price;
                end
                valid_out <= 1;
            end else begin
                valid_out <= 0;
            end
        end
    end

endmodule
