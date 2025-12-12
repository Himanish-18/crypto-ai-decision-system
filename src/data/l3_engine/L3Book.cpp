
#include <iostream>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>

// v31 Institutional L3 Order Book
// Supports Add, Cancel, Modify, Execute at O(1) or O(log N)

struct Order {
    uint64_t id;
    double price;
    double size;
    char side; // 'B' or 'S'
};

class L3Book {
private:
    // Price -> Size (L2 View)
    std::map<double, double, std::greater<double>> bids; // Descending
    std::map<double, double, std::less<double>> asks;    // Ascending
    
    // OrderID -> Order Details (L3 View)
    std::unordered_map<uint64_t, Order> orders;
    
    std::mutex book_mutex;

public:
    void add_order(uint64_t id, char side, double price, double size) {
        std::lock_guard<std::mutex> lock(book_mutex);
        
        Order ord = {id, price, size, side};
        orders[id] = ord;
        
        if (side == 'B') {
            bids[price] += size;
        } else {
            asks[price] += size;
        }
    }

    void cancel_order(uint64_t id) {
        std::lock_guard<std::mutex> lock(book_mutex);
        
        if (orders.find(id) == orders.end()) return;
        
        Order& ord = orders[id];
        if (ord.side == 'B') {
            bids[ord.price] -= ord.size;
            if (bids[ord.price] <= 1e-9) bids.erase(ord.price);
        } else {
            asks[ord.price] -= ord.size;
            if (asks[ord.price] <= 1e-9) asks.erase(ord.price);
        }
        orders.erase(id);
    }
    
    void execute_order(uint64_t id, double filled_size) {
        std::lock_guard<std::mutex> lock(book_mutex);
        
        if (orders.find(id) == orders.end()) return;
        
        Order& ord = orders[id];
        ord.size -= filled_size;
        
        if (ord.side == 'B') {
            bids[ord.price] -= filled_size;
             if (bids[ord.price] <= 1e-9) bids.erase(ord.price);
        } else {
            asks[ord.price] -= filled_size;
             if (asks[ord.price] <= 1e-9) asks.erase(ord.price);
        }
        
        if (ord.size <= 1e-9) {
            orders.erase(id);
        }
    }

    double get_best_bid() {
        std::lock_guard<std::mutex> lock(book_mutex);
        if (bids.empty()) return 0.0;
        return bids.begin()->first;
    }

    double get_best_ask() {
        std::lock_guard<std::mutex> lock(book_mutex);
        if (asks.empty()) return 0.0;
        return asks.begin()->first;
    }
};

extern "C" {
    L3Book* l3_create() { return new L3Book(); }
    void l3_free(L3Book* book) { delete book; }
    void l3_add(L3Book* book, uint64_t id, char side, double price, double size) { book->add_order(id, side, price, size); }
    void l3_cancel(L3Book* book, uint64_t id) { book->cancel_order(id); }
    double l3_get_bid(L3Book* book) { return book->get_best_bid(); }
    double l3_get_ask(L3Book* book) { return book->get_best_ask(); }
}
