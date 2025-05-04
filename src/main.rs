mod io; mod preprocess; mod model;
use io::load_csv; use preprocess::{preprocess,make_weight_driven_data}; use model::train_model;
use std::error::Error;

fn main()->Result<(),Box<dyn Error>>{
    let data=make_weight_driven_data("/Users/haoz/Desktop/IvanZ/BU/sophomore2/DS210/DS210 Final Project/ufc-fighters-statistics.csv")?;
    let coefs=train_model(&data)?;
    println!("Feature importance (by abs coeff):");
    for(n,c)in coefs{println!("{} => {:.4}",n,c);} Ok(())
}
