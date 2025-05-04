// src/model.rs
//! Module for training regression models on cleaned UFC data.
//! Provides a linear regression implementation that includes one-hot, absolute, and ratio features.

use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array2, Array1};
use crate::preprocess::{CleanRecord, WeightClass};
use std::error::Error;

/// Train a linear regression model on CleanRecord data and rank features by coefficient magnitude.
///
/// # Inputs
/// - `records`: slice of preprocessed `CleanRecord` rows.
///
/// # Outputs
/// - `Ok(Vec<(String, f64)>)`: sorted list of (feature_name, coefficient) pairs by descending |coefficient|.
/// - `Err`: if model training fails.
pub fn train_model(
    records: &[CleanRecord]
) -> Result<Vec<(String, f64)>, Box<dyn Error>> {
    let n = records.len();
    // Features: 2 stance flags + 7 weight-class flags + 10 numeric = 19
    let p = 19;

    // Build feature matrix X (n × p) and target vector y (n)
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);

    for (i, r) in records.iter().enumerate() {
        // 1) Stance one-hot (Orthodox baseline)
        x[[i, 0]] = r.is_southpaw as f64;
        x[[i, 1]] = r.is_switch   as f64;
        // 2) Weight class one-hot (Flyweight baseline)
        x[[i, 2]] = (r.weight_class == WeightClass::Bantamweight    ) as u8 as f64;
        x[[i, 3]] = (r.weight_class == WeightClass::Featherweight   ) as u8 as f64;
        x[[i, 4]] = (r.weight_class == WeightClass::Lightweight     ) as u8 as f64;
        x[[i, 5]] = (r.weight_class == WeightClass::Welterweight    ) as u8 as f64;
        x[[i, 6]] = (r.weight_class == WeightClass::Middleweight    ) as u8 as f64;
        x[[i, 7]] = (r.weight_class == WeightClass::LightHeavyweight) as u8 as f64;
        x[[i, 8]] = (r.weight_class == WeightClass::Heavyweight     ) as u8 as f64;
        // 3) Numeric features
        x[[i, 9 ]] = r.weight_height_ratio     as f64;
        x[[i,10]] = r.reach_height_ratio      as f64;
        x[[i,11]] = r.submission_per_takedown as f64;
        x[[i,12]] = r.age                     as f64;
        x[[i,13]] = r.significant_strikes_lpm as f64;
        x[[i,14]] = r.strike_diff             as f64;
        x[[i,15]] = r.takedown_lpm            as f64;
        x[[i,16]] = r.submission_lpm          as f64;
        x[[i,17]] = r.takedown_accuracy       as f64;
        x[[i,18]] = r.takedown_defense        as f64;
        // 4) Target variable
        y[i] = r.win_rate as f64;
    }

    // Fit linear regression with intercept
    let dataset = Dataset::new(x, y);
    let model = LinearRegression::default().fit(&dataset)?;

    // Extract and sort coefficients by absolute value
    let coefs = model.params();
    let feature_names = [
        // stance
        "is_southpaw", "is_switch",
        // weight class
        "wc_bantamweight", "wc_featherweight", "wc_lightweight",
        "wc_welterweight", "wc_middleweight", "wc_light_heavyweight",
        "wc_heavyweight",
        // numeric
        "weight_height_ratio", "reach_height_ratio", "submission_per_takedown",
        "age", "significant_strikes_lpm", "strike_diff",
        "takedown_lpm", "submission_lpm", "takedown_accuracy", "takedown_defense",
    ];
    let mut results: Vec<(String, f64)> = feature_names
        .iter()
        .zip(coefs.iter())
        .map(|(&name, &coef)| (name.to_string(), coef))
        .collect();
    results.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    Ok(results)
}