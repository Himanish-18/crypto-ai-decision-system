
#include <iostream>
#include <vector>
#include <atomic>

// v31 UDP Sequencer / Gap Detector
// Tracks packet sequence numbers and reports drops.

class Sequencer {
private:
    uint64_t next_expected_seq;
    uint64_t dropped_packets;

public:
    Sequencer() : next_expected_seq(0), dropped_packets(0) {}

    bool process_packet(uint64_t seq_num) {
        if (next_expected_seq == 0) {
            next_expected_seq = seq_num + 1;
            return true;
        }

        if (seq_num == next_expected_seq) {
            next_expected_seq++;
            return true;
        } else if (seq_num > next_expected_seq) {
            uint64_t gap = seq_num - next_expected_seq;
            dropped_packets += gap;
            // In a real system, we might request replay here.
            // For now, fast forward.
            next_expected_seq = seq_num + 1;
            return false; // Gap Detected
        } else {
            // Duplicate or Old packet
            return false;
        }
    }
    
    uint64_t get_drops() const { return dropped_packets; }
};

extern "C" {
    Sequencer* seq_create() { return new Sequencer(); }
    void seq_free(Sequencer* seq) { delete seq; }
    bool seq_check(Sequencer* seq, uint64_t num) { return seq->process_packet(num); }
    uint64_t seq_get_drops(Sequencer* seq) { return seq->get_drops(); }
}
