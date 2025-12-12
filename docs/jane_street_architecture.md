
# Jane Street / Citadel Style Architecture (v38)

## 1. Data Plane (Nanosecond Latency)
*   **L3 Market Data Feeds**: FPGA-based Line Handlers (Xilinx Alveo).
*   **Switching**: Arista 7130 (Layer 1 Switching).
*   **Kernel Bypass**: Solarflare OpenOnload (Userspace TCP/UDP).

## 2. Strategy Plane (Microsecond Execution)
*   **HFT Kernels**: Rust/C++ Logic (Inventory, Microstructure).
*   **Orchestration**: Python (non-critical path).
*   **Colocation**: Equinix NY4 (New Jersey) & LD4 (London).

## 3. Execution Plane
*   **Order Entry**: FPGA FIX Engine (Hardware encoding).
*   **Risk**: Pre-trade risk checks in FPGA (<100ns).
*   **OMS**: State-machine based Parent/Child order management.

## 4. Research Plane
*   **Offline Training**: GPU Cluster (A100s) for Deep RL.
*   **Simulation**: Event-Driven Packet-Level Replay (TB of pcap/day).
*   **Alpha**: Component-based Alpha Library (Mean Rev, Momentum, StatArb).

## 5. Custody & Settlement
*   **Fireblocks**: MPC Signing for withdrawals.
*   **Copper ClearLoop**: Off-exchange settlement.

![Diagram Placeholder](jane_street_diagram.png)
