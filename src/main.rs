mod io; mod preprocess; mod model;
use io::load_csv; use preprocess::{preprocess,make_weight_driven_data}; use model::train_model;
use std::error::Error;

fn main()->Result<(),Box<dyn Error>>{
    let data=make_weight_driven_data("/Users/haoz/Desktop/IvanZ/BU/sophomore2/DS210/DS210 Final Project/ufc-fighters-statistics.csv")?;
    let coefs=train_model(&data)?;
    println!("Feature importance (by abs coeff):");
    for(n,c)in coefs{println!("{} => {:.4}",n,c);} Ok(())
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
