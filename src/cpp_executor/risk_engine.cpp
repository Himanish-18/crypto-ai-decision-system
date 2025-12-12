#include <iostream>
#include <atomic>
#include <vector>
#include <string>
#include <cmath>

// C++20 standard
// Ultra-low latency risk checks

extern "C" {

    struct RiskConfig {
        double max_position;
        double max_drawdown;
        double max_risk_per_trade;
    };

    struct TradeSignal {
        double price;
        double quantity;
        int side; // 1 = Buy, -1 = Sell
    };

    class AtomicRiskEngine {
        std::atomic<double> current_exposure;
        std::atomic<double> max_drawdown_limit;
        std::atomic<double> daily_pnl;

    public:
        AtomicRiskEngine(double limit, double dd_limit) : current_exposure(0.0), max_drawdown_limit(dd_limit), daily_pnl(0.0) {}

        bool check_trade(const TradeSignal& signal) {
            // 1. Exposure Check
            double new_exposure = current_exposure.load() + (signal.quantity * signal.price);
            if (std::abs(new_exposure) > 100000.0) { // Hardcoded 100k limit for now, should use config
                return false; 
            }

            // 2. Drawdown Check
            if (daily_pnl.load() < -max_drawdown_limit.load()) {
                return false;
            }

            return true;
        }

        void update_exposure(double qty, double price) {
            double current = current_exposure.load();
            while (!current_exposure.compare_exchange_weak(current, current + (qty * price))) {
                // Spin
            }
        }
        
        void update_pnl(double pnl) {
            double current = daily_pnl.load();
            while (!daily_pnl.compare_exchange_weak(current, current + pnl)) {
                // Spin
            }
        }
    };

    // Global Risk Instance (Singleton pattern for C bindings)
    AtomicRiskEngine* engine_instance = nullptr;

    void init_risk_engine(double max_exp, double max_dd) {
        if (engine_instance) delete engine_instance;
        engine_instance = new AtomicRiskEngine(max_exp, max_dd);
    }

    bool check_risk(double price, double qty, int side) {
        if (!engine_instance) return false;
        TradeSignal sig = {price, qty, side};
        return engine_instance->check_trade(sig);
    }

    void update_position(double price, double qty) {
        if (engine_instance) engine_instance->update_exposure(qty, price);
    }
}
