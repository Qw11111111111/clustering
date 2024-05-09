use std::num::NonZeroIsize;

use crate::cluster_algos::lloyd::Kmeans;
use crate::utils::mathfuncs::{silhouette_score, create_square, center_scale};
use crate::utils::utility::*;
use plots::{line_plot, scatter_plot};
use plotters::prelude::*;
use ndarray::{Array, Array2, Axis};
use rand::prelude::*;
pub mod cluster_algos;
pub mod utils;
pub mod plots;

fn main() {
    // need to cluster data somehow... Also need a plot...
    //let size = 1000;
    //let dim = 2;
    //let bias = 42.0;
    //let weight = 2.0;
    //let x = x_data_gen(size, dim);
    //let y = get_y_data(x.clone(), &weight, &bias);
    let cluster_size = 150;
    let noise_intensity = 0;

    
    let square_1: Array2<f32> = create_square(vec![2.0, 3.0], cluster_size, 2); // Cluster 1
    let square_2: Array2<f32> = create_square(vec![7.0, 9.0], cluster_size, 2); // Cluster 2
    let square_3: Array2<f32> = create_square(vec![2.0, 3.0], cluster_size, 2); // Cluster 3
    let square_4: Array2<f32> = create_square(vec![7.0, 10.0], cluster_size + noise_intensity, 2); // A bunch of noise across them all

    let mut data: Array2<f32> = ndarray::concatenate(
        Axis(0),
        &[
            square_1.view(),
            square_2.view(),
            square_3.view(),
            square_4.view(),
        ],
    )
    .expect("An error occurred while stacking the dataset");

    center_scale(&mut data);


    let mut model = Kmeans  {
        centers: 0,
        accept: 0.7,
        max_centers: 7,
        initializer: "kmeans++",
        centroids: Array::<f32, _>::zeros((1,1)),
        partition: vec![0; cluster_size * 4 + noise_intensity],
        max_iter: 3000,
        retries: 10
    };

    let partitions = model.fit_predict(&data);
    
    println!("models is fitted");
    print_array(&model.centroids);
    println!("");
    //print_vec(&partitions);
    //println!("");
    let _ = scatter_plot("kmeans_fitted", &data, &partitions, &model.centroids);
    println!("plot generated");


}

fn x_data_gen(size: usize, dim: usize) -> Array2<f64> {
    let mut data = Array::<f64, _>::ones((size, dim));
    let mut index_1 = 0.0;

    for mut col in data.columns_mut() {
        let mut index = 0.0;
        for num in col.iter_mut() {
            *num = index * *num - index_1 * (*num);
            index += 1.0;
        }
        index_1 += 1.0;
    }
    return data;
}

fn get_y_data(x: Array2<f64>, weight: &f64, bias: &f64) -> Array2<f64> {
    return x * *weight + *bias;
}
