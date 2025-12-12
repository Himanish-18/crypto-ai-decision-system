// FPGA Order Handler Stub (v30)
// Target: Xilinx Alveo / Intel Stratix 10
// Objective: Parse FIX messages and trigger PCIe signals < 1us

module order_handler (
    input wire clk,
    input wire rst_n,
    input wire [63:0] fix_data_in,
    input wire valid_in,
    output reg [63:0] pcie_data_out,
    output reg valid_out
);

    // Finite State Machine for FIX Parsing
    parameter IDLE = 2'b00;
    parameter PARSE_HEADER = 2'b01;
    parameter PARSE_BODY = 2'b10;
    
    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            valid_out <= 0;
        end else begin
            case (state)
                IDLE: begin
                   if (valid_in) state <= PARSE_HEADER;
                end
                PARSE_HEADER: begin
                   // 8=FIX.4.2 check (stub)
                   state <= PARSE_BODY;
                end
                PARSE_BODY: begin
                   // 35=D (NewOrderSingle) trigger
                   pcie_data_out <= 64'hCAFEBABE; // Signal to CPU
                   valid_out <= 1;
                   state <= IDLE;
                end
            endcase
        end
    end

endmodule
