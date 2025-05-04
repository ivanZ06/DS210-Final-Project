// Data cleaning and statistical analysis.
use crate::io::FighterRecord;
use chrono::{Datelike, Local};
use std::str::FromStr;
use std::error::Error;

/// Fighting stance enum
#[derive(Debug, Clone, Copy)]
pub enum Stance { Orthodox, Southpaw, Switch }

impl FromStr for Stance {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Orthodox" => Ok(Stance::Orthodox),
            "Southpaw" => Ok(Stance::Southpaw),
            "Switch"   => Ok(Stance::Switch),
            _ => Err(format!("Unknown stance: {}", s)),
        }
    }
}

/// Cleaned & normalized record
#[derive(Debug)]
pub struct CleanRecord {
    pub stance: Stance,
    pub weight: f32,
    pub height: f32,
    pub reach: f32,
    pub age: f32,
    pub significant_strikes_lpm: f32,
    pub strike_diff: f32,
    pub takedown_lpm: f32,
    pub submission_lpm: f32,
    pub takedown_accuracy: f32,
    pub takedown_defense: f32,
    pub win_rate: f32,
}

/// Preprocess raw `FighterRecord` into `CleanRecord`.
pub fn preprocess(records: &[FighterRecord]) -> Vec<CleanRecord> {
    let today = Local::now().date_naive();
    let mut cleaned = Vec::new();

    for r in records {
        // unwrap or skip
        let weight = r.weight_in_kg.unwrap_or_default();
        let height = r.height_cm.unwrap_or_default();
        let reach  = r.reach_in_cm.unwrap_or_default();
        let s_lpm  = r.significant_strikes_landed_per_minute.unwrap_or_default();
        let s_abs  = r.significant_strikes_absorbed_per_minute.unwrap_or_default();
        let tkl15  = r.average_takedowns_landed_per_15_minutes.unwrap_or_default();
        let sub15  = r.average_submissions_attempted_per_15_minutes.unwrap_or_default();
        let td_acc = r.takedown_accuracy.unwrap_or_default();
        let td_def = r.takedown_defense.unwrap_or_default();

        if weight <= 0.0 { continue; }

        // parse stance
        let stance = match r.stance.parse() {
            Ok(s) => s,
            Err(_) => continue,
        };

        // compute win rate
        let total = (r.wins + r.losses + r.draws) as f32;
        let win_rate = if total > 0.0 { r.wins as f32 / total } else { 0.0 };

        // compute age
        let mut age = today.year() - r.date_of_birth.year();
        if (today.month(), today.day()) < (r.date_of_birth.month(), r.date_of_birth.day()) {
            age -= 1;
        }
        let age = age as f32;

        cleaned.push(CleanRecord {
            stance,
            weight,
            height,
            reach,
            age,
            significant_strikes_lpm: s_lpm,
            strike_diff: s_lpm - s_abs,
            takedown_lpm: tkl15,
            submission_lpm: sub15,
            takedown_accuracy: td_acc,
            takedown_defense: td_def,
            win_rate,
        });
    }

    // normalization
    fn normalize(records: &mut [CleanRecord], get: impl Fn(&CleanRecord) -> f32, set: impl Fn(&mut CleanRecord, f32)) {
        let (min, max) = records.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mi, ma), r| {
            let v = get(r);
            (mi.min(v), ma.max(v))
        });
        let range = max - min;
        for rec in records.iter_mut() {
            let v = get(rec);
            let norm = if range > 0.0 { (v - min) / range } else { 0.0 };
            set(rec, norm);
        }
    }

    normalize(&mut cleaned, |r| r.weight, |r,v| r.weight = v);
    normalize(&mut cleaned, |r| r.height, |r,v| r.height = v);
    normalize(&mut cleaned, |r| r.reach, |r,v| r.reach = v);
    normalize(&mut cleaned, |r| r.age, |r,v| r.age = v);
    normalize(&mut cleaned, |r| r.significant_strikes_lpm, |r,v| r.significant_strikes_lpm = v);
    normalize(&mut cleaned, |r| r.strike_diff, |r,v| r.strike_diff = v);
    normalize(&mut cleaned, |r| r.takedown_lpm, |r,v| r.takedown_lpm = v);
    normalize(&mut cleaned, |r| r.submission_lpm, |r,v| r.submission_lpm = v);
    normalize(&mut cleaned, |r| r.takedown_accuracy, |r,v| r.takedown_accuracy = v);
    normalize(&mut cleaned, |r| r.takedown_defense, |r,v| r.takedown_defense = v);
    normalize(&mut cleaned, |r| r.win_rate, |r,v| r.win_rate = v);

    cleaned
}

/// Helper to load, preprocess and return `CleanRecord` vec.
pub fn make_weight_driven_data(path: &str) -> Result<Vec<CleanRecord>, Box<dyn Error>> {
    let raw = crate::io::load_csv(path)?;
    Ok(preprocess(&raw))
}
