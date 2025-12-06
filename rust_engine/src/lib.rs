use pyo3.prelude::*;
use polars::prelude::*;
use std::io::Cursor;

#[pyfunction]
fn calculate_features(json_data: String) -> PyResult<String> {
    // 1. Load Data
    let cursor = Cursor::new(json_data);
    let df = JsonReader::new(cursor)
        .finish()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // 2. Lazy Feature Engineering (Polars)
    let lf = df.lazy();
    
    // Example: Feature: Volatility (Rolling StdDev)
    // In real system, we re-implement all features here.
    // Simplifying for demo: Calculate RSI-like or Volatility
    
    let lf = lf.with_columns(vec![
        col("close").alias("btc_close"), // Ensure alias
    ]);

    let lf = lf.with_columns(vec![
        // Return
        col("btc_close").pct_change(lit(1)).alias("ret_1"),
    ]);
    
    // Drop Nulls
    let lf = lf.drop_nulls(None);

    // 3. Collect
    let final_df = lf.collect()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // 4. Return as JSON
    let mut buf = Vec::new();
    JsonWriter::new(&mut buf)
        .finish(&mut final_df.clone())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let json_str = String::from_utf8(buf)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(json_str)
}

#[pymodule]
fn rust_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_features, m)?)?;
    Ok(())
}
