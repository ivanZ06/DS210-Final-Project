// Module for loading and validating the data. It reads the csv file, validates headers, and handles missing data.
use std::error::Error;
use std::fs::File;
use csv::{ReaderBuilder, StringRecord};
use serde::Deserialize;
use chrono::NaiveDate;

mod date_format {
    use chrono::NaiveDate;
    use serde::{self, Deserialize, Deserializer};
    const FMT: &str = "%Y-%m-%d";

    pub fn deserialize<'de, D>(d: D) -> Result<NaiveDate, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(d)?;
        NaiveDate::parse_from_str(&s, FMT).map_err(serde::de::Error::custom)
    }
}

/// Matches exactly your 18 CSV columns.
#[derive(Debug, Deserialize)]
pub struct FighterRecord {
    #[serde(rename = "name")]   pub name: String,
    #[serde(rename = "nickname")]   pub nickname: Option<String>,
    #[serde(rename = "wins")]   pub wins: usize,
    #[serde(rename = "losses")] pub losses: usize,
    #[serde(rename = "draws")]  pub draws: usize,
    #[serde(rename = "height_cm")]  pub height_cm: Option<f32>,
    #[serde(rename = "weight_in_kg")] pub weight_in_kg: Option<f32>,
    #[serde(rename = "reach_in_cm")]  pub reach_in_cm: Option<f32>,
    #[serde(rename = "stance")]      pub stance: String,
    #[serde(rename = "date_of_birth", deserialize_with = "date_format::deserialize")]
                                      pub date_of_birth: NaiveDate,
    #[serde(rename = "significant_strikes_landed_per_minute")]
                                      pub significant_strikes_landed_per_minute: Option<f32>,
    #[serde(rename = "significant_striking_accuracy")]
                                      pub significant_striking_accuracy: Option<f32>,
    #[serde(rename = "significant_strikes_absorbed_per_minute")]
                                      pub significant_strikes_absorbed_per_minute: Option<f32>,
    #[serde(rename = "significant_strike_defence")]
                                      pub significant_strike_defence: Option<f32>,
    #[serde(rename = "average_takedowns_landed_per_15_minutes")]
                                      pub average_takedowns_landed_per_15_minutes: Option<f32>,
    #[serde(rename = "takedown_accuracy")]    pub takedown_accuracy: Option<f32>,
    #[serde(rename = "takedown_defense")]     pub takedown_defense: Option<f32>,
    #[serde(rename = "average_submissions_attempted_per_15_minutes")]
                                      pub average_submissions_attempted_per_15_minutes: Option<f32>,
}

pub fn load_csv(path: &str) -> Result<Vec<FighterRecord>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b',')
        .flexible(true)
        .has_headers(true)
        .from_reader(file);

    // Grab and own the header row
    let headers = rdr.headers()?.clone();
    let expected_len = headers.len();

    let mut out = Vec::new();
    for result in rdr.records() {
        let raw: StringRecord = result?;

        // 1) Skip completely empty lines
        if raw.iter().all(|f| f.trim().is_empty()) {
            continue;
        }

        // 2) Skip rows with the wrong number of fields
        if raw.len() != expected_len {
            eprintln!(
                "Skipping line {}: expected {} fields, found {}",
                // Here’s the fix: use `map` instead of `and_then`
                raw.position().map(|p| p.line()).unwrap_or(0),
                expected_len,
                raw.len(),
            );
            continue;
        }

        // 3) Attempt to deserialize; if it fails, skip that row
        match raw.deserialize::<FighterRecord>(Some(&headers)) {
            Ok(rec) => out.push(rec),
            Err(e) => {
                eprintln!(
                    "Skipping malformed record at line {}: {}",
                    raw.position().map(|p| p.line()).unwrap_or(0),
                    e
                );
            }
        }
    }

    Ok(out)
}
