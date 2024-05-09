use std::{thread::panicking, vec};
use num::ToPrimitive;
use plotters::{prelude::*, style::full_palette::CYAN_A700};
use ndarray::{array, Array, Array1, Array2};
use crate::utils::mathfuncs::*;
use crate::utils::utility::*;

pub fn line_plot(x: &Array2<f64>, y: &Array1<f64>, pred: &Array1<f64>, name: String) -> Result<(), Box<dyn std::error::Error>> {

    let mut x_ = Array::from_vec(vec![0.0; x.len()]);
    let mut path = String::from("./images/");

    path.push_str(&name);
    path.push_str(".png");

    for (i, row) in x.rows().into_iter().enumerate() {
        x_[i] = row[0];
    } 

    let predictions = create_vec(&x_, pred);
    let true_vals = create_vec(&x_, y);

    let pred_series = LineSeries::new(predictions, &RED);
    let true_series = LineSeries::new(true_vals, &BLACK);

    let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(10, 10, 10, 10);

    let mut chart = ChartBuilder::on(&root)
        .caption("plot", ("sans-serif", 40).into_font())
        .x_label_area_size(10)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..10000.0, 0.0..1500.0)?;

    chart.configure_mesh().x_labels(10).y_labels(10).draw()?;

    chart.draw_series(pred_series)?;
    chart.draw_series(true_series)?;

    root.present()?;
    Ok(())
}

pub fn scatter_plot(name: &str, data: &Array2<f32>, partitions: &Vec<i32>, centroids: &Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {

    let mut path = String::from("./images/");

    let max = max_int(partitions.to_owned());

    // for max 7 clusters
    let styles = vec![&BLACK, &RED, &MAGENTA, &GREEN, &BLUE, &CYAN, &CYAN_A700];

    path.push_str(name);
    path.push_str(".png");
    
    let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(10, 10, 10, 10);

    let mut chart = ChartBuilder::on(&root)
        .caption("plot", ("sans-serif", 40).into_font())
        .x_label_area_size(10)
        .y_label_area_size(40)
        .build_cartesian_2d(-5.0..5.0, -5.0..5.0)?;

    chart.configure_mesh().x_labels(10).y_labels(10).draw()?;

    for i in 0..max + 1{
        let mut series_data: Vec<(f64, f64)> = vec![(0.0, 0.0)];
        for j in 0..partitions.len() {
            if partitions[j] == i {
                //assuming 2D input. This makes sense, as higher D inouts needs different plot.
                series_data.push((data[[j, 0]].to_f64().unwrap(), data[[j, 1]].to_f64().unwrap()));
            };
        }
        series_data.remove(0);
        if series_data.is_empty(){
            continue;
        }
        //let series: PointSeries<(f64, f64), vec::IntoIter<(f64, f64)>, _, i32> = PointSeries::new(series_data.into_iter(), 2, styles[i.to_usize().unwrap()].filled());
        chart.draw_series(
            series_data
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, styles[i.to_usize().unwrap()].filled())),
        )?;
    }

    for i in 0..max.to_usize().unwrap() + 1{
        let series_data: Vec<(f64, f64)> = vec![(centroids[[i, 0]].to_f64().unwrap(), centroids[[i, 1]].to_f64().unwrap())];
        chart.draw_series(
            series_data
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, (&YELLOW).filled())),
        )?;
    }

    root.present()?;
    Ok(())
}



fn create_vec(data: &Array1<f64>, y: &Array1<f64>) -> Vec<(f64, f64)> {
    let mut predictions = vec![(0.0, 0.0); data.len()];

    for (i, num) in data.iter().enumerate() {
        predictions[i].0 = *num;
        predictions[i].1 = y[i];
    }

    return predictions;
}