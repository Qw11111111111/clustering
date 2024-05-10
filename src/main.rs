use std::collections::HashSet;
use std::num::NonZeroIsize;
use crate::cluster_algos::agglomerative::AgglomerativeCluster;
use crate::cluster_algos::dbscan::DBScan;
use crate::cluster_algos::lloyd::Kmeans;
use crate::utils::mathfuncs::{silhouette_score, create_square, center_scale};
use crate::utils::utility::*;
use plots::{line_plot, scatter_plot};
use plotters::prelude::*;
use ndarray::{array, Array, Array2, Axis};
use rand::prelude::*;
pub mod cluster_algos;
pub mod utils;
pub mod plots;

fn main() {
    let cluster_size = 300;
    let noise_intensity = 0;
    let num_clusters = 4;

    let mut data = get_data(noise_intensity, num_clusters, cluster_size);

    center_scale(&mut data);
    
    let mut model = DBScan {
        min_points: 20,
        epsilon: 15e-2,
        is_in_cluster: HashSet::new(),
        is_visited: HashSet::new(),
        is_noise: HashSet::new(),
        partitions: vec![0],
        current_clusters: 1
    };
    
    /* 
    let mut model = AgglomerativeCluster {
        centers: num_clusters,
        clusters: vec![vec![array![0.0]]]
    };
    */

    /*
    let mut model = Kmeans  {
        centers: num_clusters,
        accept: 0.7,
        max_centers: 7,
        initializer: "kmeans++",
        centroids: Array::<f32, _>::zeros((1,1)),
        partition: vec![0; cluster_size * 4 + noise_intensity],
        max_iter: 3000,
        retries: 10
    };
    */
    let partitions = model.fit_predict(&data);
    print_vec(&partitions);
    println!("models is fitted");
    //print_array(&model.centroids);
    println!("");
    //print_vec(&partitions);
    let centroids = array![[0.0, 0.0]];
    //println!("");
    let _ = scatter_plot("DBScan_fitted", &data, &partitions, &centroids);
    println!("plot generated");


}


fn get_data(noise_intensity: usize, num_clusters: i32, cluster_size: usize) -> Array2<f32> {
    let square_1: Array2<f32> = create_square(vec![1.0, 3.0], vec![2.0, 4.0], cluster_size, 2); // Cluster 1
    let square_2: Array2<f32> = create_square(vec![5.0, 7.0], vec![1.0, 3.0], cluster_size, 2); // Cluster 2
    let square_3: Array2<f32> = create_square(vec![5.0, 7.0], vec![6.0, 7.0], cluster_size, 2); // Cluster 3
    let square_4: Array2<f32> = create_square(vec![10.0, 12.0], vec![6.0, 7.0], cluster_size, 2);
    let square_5: Array2<f32> = create_square(vec![1.0, 8.0], vec![1.0, 7.0], cluster_size / 10 + noise_intensity, 2); // A bunch of noise across them all

    let data: Array2<f32> = ndarray::concatenate(
        Axis(0),
        &[
            square_1.view(),
            square_2.view(),
            square_3.view(),
            square_4.view(),
            square_5.view()
        ],
    )
    .expect("An error occurred while stacking the dataset");

    return data;
}