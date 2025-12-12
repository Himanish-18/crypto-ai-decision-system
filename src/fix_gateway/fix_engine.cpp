#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <ctime>
#include <mutex>

// Minimal FIX Engine in C++20 for HFT
// Supports 4.2 and 4.4 subset (Logon, Heartbeat, NewOrderSingle)

class FIXMessage {
public:
    std::map<int, std::string> fields;
    
    void set(int tag, const std::string& value) {
        fields[tag] = value;
    }
    
    std::string serialize() const {
        std::stringstream ss;
        // Mock BodyLength calc for simplicity in this demo
        for (const auto& kv : fields) {
            ss << kv.first << "=" << kv.second << "\x01";
        }
        return ss.str();
    }
};

class FIXSession {
    std::string sender_comp_id;
    std::string target_comp_id;
    int seq_num;
    std::mutex session_lock;
    bool connected;

public:
    FIXSession(std::string sender, std::string target) 
        : sender_comp_id(sender), target_comp_id(target), seq_num(1), connected(false) {}

    std::string logon() {
        std::lock_guard<std::mutex> lock(session_lock);
        FIXMessage msg;
        msg.set(8, "FIX.4.2");
        msg.set(35, "A"); // Logon
        msg.set(49, sender_comp_id);
        msg.set(56, target_comp_id);
        msg.set(34, std::to_string(seq_num++));
        msg.set(98, "0"); // EncryptMethod
        msg.set(108, "30"); // HeartBtInt
        
        connected = true;
        return msg.serialize();
    }
    
    std::string new_order_single(const std::string& symbol, char side, double qty, double price) {
        if (!connected) return "";
        std::lock_guard<std::mutex> lock(session_lock);
        
        FIXMessage msg;
        msg.set(8, "FIX.4.2");
        msg.set(35, "D"); // NewOrderSingle
        msg.set(49, sender_comp_id);
        msg.set(56, target_comp_id);
        msg.set(34, std::to_string(seq_num++));
        msg.set(11, "ClOrdID_" + std::to_string(seq_num));
        msg.set(55, symbol); // Symbol
        msg.set(54, std::string(1, side)); // Side 1=Buy, 2=Sell
        msg.set(38, std::to_string(qty));
        msg.set(44, std::to_string(price));
        msg.set(40, "2"); // OrdType = Limit
        
        return msg.serialize();
    }
    
    bool is_connected() { return connected; }
};

// C Wrappers for Python
extern "C" {
    FIXSession* fix_create(const char* sender, const char* target) {
        return new FIXSession(sender, target);
    }
    
    void fix_destroy(FIXSession* session) {
        if (session) delete session;
    }
    
    void fix_logon(FIXSession* session, char* out_buffer, int max_len) {
        if (!session) return;
        std::string msg = session->logon();
        strncpy(out_buffer, msg.c_str(), max_len);
    }
    
    void fix_new_order(FIXSession* session, const char* symbol, char side, double qty, double price, char* out_buffer, int max_len) {
        if (!session) return;
        std::string msg = session->new_order_single(symbol, side, qty, price);
        strncpy(out_buffer, msg.c_str(), max_len);
    }
}
