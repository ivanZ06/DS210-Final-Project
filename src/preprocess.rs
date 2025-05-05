/// Data cleaning and preprocessing.

use crate::io::FighterRecord;
use chrono::{Datelike, Local};
use std::str::FromStr;
use std::error::Error;

/// Represents different fighting stances; used to represent the stances
#[derive(Debug, Clone, Copy)]
pub enum Stance {
    Orthodox,
    Southpaw,
    Switch,
}

/// Convert a raw stance string into the corresponding “Stance” enum variant
/// input: stance label from CSV
/// output: “Result<Stance, String>” – “Ok(variant)” for known labels, “Err” otherwise
/// logic: match the input string against each known case; return the matching variant or an error message
impl FromStr for Stance {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Orthodox" => Ok(Stance::Orthodox),
            "Southpaw" => Ok(Stance::Southpaw),
            "Switch"   => Ok(Stance::Switch),
            other       => Err(format!("Unknown stance: {}", other)),
        }
    }
}

/// Weight class categories
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightClass {
    Flyweight,       // <56.7 kg
    Bantamweight,    // 56.7–61.2
    Featherweight,   // 61.2–65.8
    Lightweight,     // 65.8–70.3
    Welterweight,    // 70.3–77.1
    Middleweight,    // 77.1–83.9
    LightHeavyweight,// 83.9–93.0
    Heavyweight,     // >93.0
}

/// Resulting cleaned record with engineered features (per-minute rates); used to represent the cleaned rows
#[derive(Debug)]
pub struct CleanRecord {
    pub stance: Stance,
    // One-hot stance flags
    pub is_orthodox:             f32,
    pub is_southpaw:             f32,
    pub is_switch:               f32,
    // Ratio features
    pub weight_height_ratio:     f32,
    pub reach_height_ratio:      f32,
    // Efficiency metrics
    pub submission_per_takedown: f32,
    // Weight class bucket
    pub weight_class:            WeightClass,
    // Other numeric features
    pub age:                     f32,
    pub significant_strikes_lpm: f32,
    pub strike_diff:             f32,
    pub takedown_lpm:            f32, // now per minute
    pub submission_lpm:          f32, // now per minute
    pub takedown_accuracy:       f32,
    pub takedown_defense:        f32,
    pub win_rate:                f32,
}

/// Clean raw fighter records and engineer normalized features for modeling
/// input: raw, CSV‑deserialized data
/// output: valid records with one‑hot flags, ratios, rates, and normalized numerics
/// logic: unwrap or drop invalid/NaN numeric fields; one‑hot encode stance;
/// compute win_rate, age, weight/height & reach/height ratios, per‑minute rates, efficiency, weight_class;
/// normalize all numeric features to [0,1]
pub fn preprocess(records: &[FighterRecord]) -> Vec<CleanRecord> {
    let today = Local::now().date_naive();
    let mut cleaned = Vec::new();

    for r in records {
        // 1) unwrap or default numeric inputs
        let weight = r.weight_in_kg.unwrap_or_default();
        let height = r.height_cm.unwrap_or_default();
        let reach  = r.reach_in_cm.unwrap_or_default();
        if weight <= 0.0 || height <= 0.0 { continue; }
        let s_lpm = r.significant_strikes_landed_per_minute.unwrap_or_default();
        let s_abs = r.significant_strikes_absorbed_per_minute.unwrap_or_default();
        let tkl15 = r.average_takedowns_landed_per_15_minutes.unwrap_or_default();
        let sub15 = r.average_submissions_attempted_per_15_minutes.unwrap_or_default();
        let td_acc= r.takedown_accuracy.unwrap_or_default();
        let td_def= r.takedown_defense.unwrap_or_default();
        // drop NaNs
        if [weight, height, reach, s_lpm, s_abs, tkl15, sub15, td_acc, td_def]
            .iter().any(|v| v.is_nan()) { continue; }

        // 2) parse stance + one-hot
        let stance = match r.stance.parse::<Stance>() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let (iso, isp, iss) = match stance {
            Stance::Orthodox => (1.0, 0.0, 0.0),
            Stance::Southpaw => (0.0, 1.0, 0.0),
            Stance::Switch   => (0.0, 0.0, 1.0),
        };

        // 3) compute win rate
        let total = (r.wins + r.losses + r.draws) as f32;
        let win_rate = if total > 0.0 { r.wins as f32 / total } else { 0.0 };

        // compute age
        let mut age = today.year() - r.date_of_birth.year();
        if (today.month(), today.day()) < (r.date_of_birth.month(), r.date_of_birth.day()) {
            age -= 1;
        }
        let age = age as f32;

        // 4) engineer features
        let weight_height_ratio = weight / height;
        let reach_height_ratio  = reach / height;
        // convert to per-minute rates
        let takedown_lpm   = tkl15 / 15.0;
        let submission_lpm = sub15 / 15.0;
        // efficiency metric
        let submission_per_takedown = if takedown_lpm > 0.0 { submission_lpm / takedown_lpm } else { 0.0 };
        // weight class bucket
        let weight_class = if weight < 56.7 {
            WeightClass::Flyweight
        } else if weight < 61.2 {
            WeightClass::Bantamweight
        } else if weight < 65.8 {
            WeightClass::Featherweight
        } else if weight < 70.3 {
            WeightClass::Lightweight
        } else if weight < 77.1 {
            WeightClass::Welterweight
        } else if weight < 83.9 {
            WeightClass::Middleweight
        } else if weight < 93.0 {
            WeightClass::LightHeavyweight
        } else {
            WeightClass::Heavyweight
        };

        cleaned.push(CleanRecord {
            stance,
            is_orthodox:             iso,
            is_southpaw:             isp,
            is_switch:               iss,
            weight_height_ratio,
            reach_height_ratio,
            submission_per_takedown,
            weight_class,
            age,
            significant_strikes_lpm: s_lpm,
            strike_diff:             s_lpm - s_abs,
            takedown_lpm,
            submission_lpm,
            takedown_accuracy:       td_acc,
            takedown_defense:        td_def,
            win_rate,
        });
    }

    // 5) normalize all numeric fields to [0,1]
    /// Inputs: "records": mutable slice of `CleanRecord` to modify in place,
    /// "getter": function to extract the raw feature value from a record,
    /// "setter": function to assign the normalized value back into the record.
    /// logic: Fold over all records to find the feature’s global `min` and `max`;
    /// Loop through each record, compute "(value - min) / (max - min)"" (0 if min == max) and set it.
    fn normalize(
        records: &mut [CleanRecord],
        getter: impl Fn(&CleanRecord) -> f32,
        setter: impl Fn(&mut CleanRecord, f32),
    ) {
        // 1) Compute global min and max for this feature
        let (min, max) = records.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(mi, ma), r| {
                let v = getter(r);
                (mi.min(v), ma.max(v))
            },
        );
        let range = max - min;

         // 2) Normalize each record in place
        for rec in records.iter_mut() {
            let v = getter(rec);
            // avoid division by zero when all values are equal
            let norm = if range > 0.0 { (v - min) / range } else { 0.0 };
            setter(rec, norm);
        }
    }

    // Apply normalization to each engineered feature
    normalize(&mut cleaned, |r| r.weight_height_ratio,      |r,v| r.weight_height_ratio = v);
    normalize(&mut cleaned, |r| r.reach_height_ratio,       |r,v| r.reach_height_ratio = v);
    normalize(&mut cleaned, |r| r.submission_per_takedown,  |r,v| r.submission_per_takedown = v);
    normalize(&mut cleaned, |r| r.age,                      |r,v| r.age = v);
    normalize(&mut cleaned, |r| r.significant_strikes_lpm,  |r,v| r.significant_strikes_lpm = v);
    normalize(&mut cleaned, |r| r.strike_diff,              |r,v| r.strike_diff = v);
    normalize(&mut cleaned, |r| r.takedown_lpm,             |r,v| r.takedown_lpm = v);
    normalize(&mut cleaned, |r| r.submission_lpm,           |r,v| r.submission_lpm = v);
    normalize(&mut cleaned, |r| r.takedown_accuracy,        |r,v| r.takedown_accuracy = v);
    normalize(&mut cleaned, |r| r.takedown_defense,         |r,v| r.takedown_defense = v);
    normalize(&mut cleaned, |r| r.win_rate,                 |r,v| r.win_rate = v);

    cleaned
}

/// Load raw CSV and run preprocessing to produce cleaned records
/// input: filesystem path to the fighters CSV
/// output: cleaned and normalized data or error
/// logic: call "io::load_csv(path)" to parse raw records, then "preprocess(&raw)" to engineer features
pub fn make_weight_driven_data(path: &str) -> Result<Vec<CleanRecord>, Box<dyn Error>> {
    let raw = crate::io::load_csv(path)?;
    Ok(preprocess(&raw))
}
