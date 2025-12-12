use std::ffi::CStr;
use std::os::raw::c_char;
use std::collections::HashMap;

#[no_mangle]
pub extern "C" fn calculate_microprice(bid_price: f64, ask_price: f64, bid_qty: f64, ask_qty: f64) -> f64 {
    let total_qty = bid_qty + ask_qty;
    if total_qty == 0.0 {
        return (bid_price + ask_price) / 2.0;
    }
    (bid_price * ask_qty + ask_price * bid_qty) / total_qty
}

#[no_mangle]
pub extern "C" fn calculate_imbalance(bid_vol: f64, ask_vol: f64) -> f64 {
    let total = bid_vol + ask_vol;
    if total == 0.0 {
        return 0.0;
    }
    (bid_vol - ask_vol) / total
}

struct OrderBook {
    bids: HashMap<String, f64>,
    asks: HashMap<String, f64>,
}

impl OrderBook {
    fn new() -> Self {
        OrderBook {
            bids: HashMap::new(),
            asks: HashMap::new(),
        }
    }
}

#[no_mangle]
pub extern "C" fn orderbook_new() -> *mut OrderBook {
    Box::into_raw(Box::new(OrderBook::new()))
}

#[no_mangle]
pub extern "C" fn orderbook_free(ptr: *mut OrderBook) {
    if ptr.is_null() { return; }
    unsafe { 
        let _ = Box::from_raw(ptr); 
    }
}

#[no_mangle]
pub extern "C" fn orderbook_update_bid(ptr: *mut OrderBook, price: *const c_char, qty: f64) {
    let ob = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    let c_str = unsafe { CStr::from_ptr(price) };
    let p_str = c_str.to_string_lossy().into_owned();
    
    if qty <= 0.0 {
        ob.bids.remove(&p_str);
    } else {
        ob.bids.insert(p_str, qty);
    }
}

#[no_mangle]
pub extern "C" fn orderbook_update_ask(ptr: *mut OrderBook, price: *const c_char, qty: f64) {
    let ob = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    let c_str = unsafe { CStr::from_ptr(price) };
    let p_str = c_str.to_string_lossy().into_owned();
    
    if qty <= 0.0 {
        ob.asks.remove(&p_str);
    } else {
        ob.asks.insert(p_str, qty);
    }
}

#[no_mangle]
pub extern "C" fn orderbook_get_best_bid(ptr: *mut OrderBook, out_price: *mut f64, out_qty: *mut f64) {
    let ob = unsafe { &*ptr };
    let mut best_price = 0.0;
    let mut best_qty = 0.0;
    
    for (p_str, &q) in &ob.bids {
        let p: f64 = p_str.parse().unwrap_or(0.0);
        if p > best_price {
            best_price = p;
            best_qty = q;
        }
    }
    unsafe {
        *out_price = best_price;
        *out_qty = best_qty;
    }
}
