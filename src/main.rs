/// Put all the codes from the modules together and run them to give the output and visualization
use std::error::Error;
use std::env;

mod io;
mod preprocess;
mod model;

use io::load_csv;
use preprocess::preprocess;
use model::train_model;
use plotters::prelude::*;

/// Draws a horizontal bar chart of feature importances and saves it as “feature_importances.png”
/// input: feature names with their coefficients  
/// output: none (saves "feature_importances.png" to the current directory)
/// logic: split "results" into names and values; compute X‐axis range; set up PNG backend; 
/// build Cartesian chart; label Y ticks with feature names; draw one bar per coefficient  
fn plot_importances(results: &[(String, f64)]) -> Result<(), Box<dyn std::error::Error>> {
    // Split into names and coefficients
    let names: Vec<&str> = results.iter().map(|(n, _)| n.as_str()).collect();
    let coefs: Vec<f64> = results.iter().map(|(_, c)| *c).collect();
    // Use the total number of results for the Y-axis range
    let count = results.len();

    // Determine X axis range with padding
    let min_x = coefs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = coefs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pad = (max_x - min_x) * 0.1;
    let x_range = (min_x - pad)..(max_x + pad);

    // Prepare drawing area
    let root = BitMapBackend::new("feature_importances.png", (1000, 600))
        .into_drawing_area();
    root.fill(&WHITE)?;

    // Build chart with numeric X and integer Y covering all features
    let mut chart = ChartBuilder::on(&root)
        .caption("Feature Importances", ("sans-serif", 24))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(200)
        .build_cartesian_2d(x_range, 0..count)?;

    // Configure mesh: label each Y tick with its feature name
    chart
        .configure_mesh()
        .disable_mesh()
        // Show one label per feature; hide any out‑of‑range ticks
        .y_labels(count)
        .y_label_formatter(&|idx| {
            let i = *idx as usize;
            if i < count {
                names[i].to_string()
            } else {
                String::new()
            }
        })
        .x_desc("Coefficient")
        .y_desc("Feature")
        .draw()?;

    // Draw horizontal bars for each feature index 0..count
    chart.draw_series(
        coefs.iter().enumerate().map(|(i, &coef)| {
            let start = 0.0_f64.min(coef);
            let end = 0.0_f64.max(coef);
            Rectangle::new([(start, i), (end, i + 1)], BLUE.mix(0.5).filled())
        })
    )?;

    Ok(())
}

/// load data, preprocess, model training, and visualize
/// input: none
/// output: none
/// logic: Parse CLI argument for CSV path; Call "load_csv"; Call "preprocess(&raw)";
/// Call "train_model(&cleaned)";Print each "(feature, coefficient) to stdout;
/// Call “plot_importances(&results)” to save a bar‑chart PNG  
fn main() -> Result<(), Box<dyn Error>> {
    // 1) Read CSV path
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "ufc-fighters-statistics.csv".into());
    println!("Loading data from {}...", path);

    // 2) Load and preprocess
    let raw = load_csv(&path)?;
    let cleaned = preprocess(&raw);
    println!("Processed {} records", cleaned.len());

    // 3) Train model
    let results = train_model(&cleaned)?;
    println!("\nFeature importances:");
    for (name, coef) in &results {
        println!("{:<30} {:>8.4}", name, coef);
    }

    // 4) Plot and save to PNG
    plot_importances(&results)?;
    println!("Wrote feature_importances.png");

    Ok(())
}

/// the test functions
#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs::File, io::Write};
    use std::error::Error;
    use chrono::NaiveDate;

    // for IO tests
    use crate::io::load_csv;
    // for preprocessing tests
    use crate::preprocess::{preprocess, CleanRecord, Stance, WeightClass};
    // for model tests
    use crate::model::train_model;

    /// IO: can read a single well‑formed record
    #[test]
    fn test_load_csv() -> Result<(), Box<dyn Error>> {
        let path = "test_fighters.csv";
        let mut f = File::create(path)?;
        writeln!(&mut f, concat!(
            "name,nickname,wins,losses,draws,",
            "height_cm,weight_in_kg,reach_in_cm,stance,",
            "date_of_birth,",
            "significant_strikes_landed_per_minute,",
            "significant_striking_accuracy,",
            "significant_strikes_absorbed_per_minute,",
            "significant_strike_defence,",
            "average_takedowns_landed_per_15_minutes,",
            "takedown_accuracy,takedown_defense,",
            "average_submissions_attempted_per_15_minutes\n"
        ))?;
        writeln!(&mut f, concat!(
            "A,,10,2,1,",           // name,empty,n,w,l,d
            "180.0,70.0,190.0,Orthodox,", // size + stance
            "1990-01-01,",           // DOB
            "5.0,0.5,3.0,0.6,",       // strike stats
            "30.0,0.4,0.7,",          // takedown stats
            "15.0"                    // submissions per 15
        ))?;

        let recs = load_csv(path)?;
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        assert_eq!(r.name, "A");
        assert_eq!(r.nickname, None);
        assert_eq!(r.wins, 10);
        assert!((r.height_cm.unwrap() - 180.0).abs() < 1e-6);
        assert_eq!(r.stance, "Orthodox");
        assert_eq!(r.date_of_birth, NaiveDate::from_ymd(1990,1,1));
        Ok(())
    }

    /// PREPROCESS: drops bad rows, computes one‑hots + ratios + per‑minute
    /// PREPROCESS: drops bad rows and normalizes all numeric features to 0.0
    #[test]
    fn test_preprocess_filters_and_features() -> Result<(), Box<dyn Error>> {
        let path = "test_pre.csv";
        let mut f = File::create(path)?;
        writeln!(&mut f, concat!(
            "name,nickname,wins,losses,draws,",
            "height_cm,weight_in_kg,reach_in_cm,stance,",
            "date_of_birth,",
            "significant_strikes_landed_per_minute,",
            "significant_striking_accuracy,",
            "significant_strikes_absorbed_per_minute,",
            "significant_strike_defence,",
            "average_takedowns_landed_per_15_minutes,",
            "takedown_accuracy,takedown_defense,",
            "average_submissions_attempted_per_15_minutes\n"
        ))?;
        // good row
        writeln!(&mut f, concat!(
            "A,,10,2,1,180.0,90.0,190.0,Orthodox,1990-01-01,",
            "5.0,0.5,3.0,0.6,30.0,0.4,0.7,15.0"
        ))?;
        // bad row (missing weight)
        writeln!(&mut f, concat!(
            "B,,8,3,1,180.0,,190.0,Southpaw,1992-06-01,",
            "4.0,0.4,2.0,0.5,20.0,0.3,0.6,10.0"
        ))?;
    
        let raw     = load_csv(path)?;
        let cleaned = preprocess(&raw);
        assert_eq!(cleaned.len(), 1);
        let cr = &cleaned[0];
    
        // Stance one‑hots should still be correct
        assert_eq!(cr.is_orthodox, 1.0);
        assert_eq!(cr.is_southpaw, 0.0);
    
        // Because only one record survived, **all** normalized numeric features == 0.0
        assert_eq!(cr.weight_height_ratio,     0.0);
        assert_eq!(cr.reach_height_ratio,      0.0);
        assert_eq!(cr.submission_per_takedown, 0.0);
        assert_eq!(cr.takedown_lpm,            0.0);
        assert_eq!(cr.submission_lpm,          0.0);
    
        Ok(())
    }
    

    /// MODEL: simplest two‑point dataset: weight_height_ratio maps to win_rate
    #[test]
    fn test_train_model_simple() {
        let recs = vec![
            CleanRecord {
                stance: Stance::Orthodox,
                is_orthodox: 1.0, is_southpaw: 0.0, is_switch: 0.0,
                weight_height_ratio: 1.0, reach_height_ratio: 0.0,
                submission_per_takedown: 0.0, weight_class: WeightClass::Bantamweight,
                age: 0.0, significant_strikes_lpm: 0.0, strike_diff: 0.0,
                takedown_lpm: 0.0, submission_lpm: 0.0,
                takedown_accuracy: 0.0, takedown_defense: 0.0,
                win_rate: 1.0,
            },
            CleanRecord {
                stance: Stance::Orthodox,
                is_orthodox: 1.0, is_southpaw: 0.0, is_switch: 0.0,
                weight_height_ratio: 0.0, reach_height_ratio: 0.0,
                submission_per_takedown: 0.0, weight_class: WeightClass::Bantamweight,
                age: 0.0, significant_strikes_lpm: 0.0, strike_diff: 0.0,
                takedown_lpm: 0.0, submission_lpm: 0.0,
                takedown_accuracy: 0.0, takedown_defense: 0.0,
                win_rate: 0.0,
            },
        ];
        let coefs = train_model(&recs).expect("training failed");
        let wh = coefs.iter()
            .find(|&(name, _)| name == "weight_height_ratio")
            .unwrap().1;
        assert!(wh > 0.0, "Expected positive coefficient for weight_height_ratio");
    }
} // end tests
