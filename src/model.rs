/// Train linear regression models.
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array2, Array1};
use crate::preprocess::CleanRecord;
use std::error::Error;

pub fn train_model(records:&[CleanRecord])->Result<Vec<(String,f64)>,Box<dyn Error>>{
    let n=records.len(); let p=10;
    let mut x=Array2::<f64>::zeros((n,p)); let mut y=Array1::<f64>::zeros(n);
    for(i,r)in records.iter().enumerate(){
        x[(i,0)]=r.weight as f64; x[(i,1)]=r.height as f64; x[(i,2)]=r.reach as f64;
        x[(i,3)]=r.age as f64; x[(i,4)]=r.significant_strikes_lpm as f64;
        x[(i,5)]=r.strike_diff as f64; x[(i,6)]=r.takedown_lpm as f64;
        x[(i,7)]=r.submission_lpm as f64; x[(i,8)]=r.takedown_accuracy as f64;
        x[(i,9)]=r.takedown_defense as f64; y[i]=r.win_rate as f64;}
    let ds=Dataset::new(x.clone(),y.clone());
    let model=LinearRegression::new().fit(&ds)?;
    let coefs=model.params();
    let names=["weight","height","reach","age","strikes_lpm","strike_diff","takedown_lpm","submission_lpm","td_accuracy","td_defense"];
    let mut vec:Vec<(String,f64)>=names.iter().zip(coefs.iter()).map(|(&n,&c)|(n.to_string(),c)).collect();
    vec.sort_by(|a,b|b.1.abs().partial_cmp(&a.1.abs()).unwrap()); Ok(vec)}
