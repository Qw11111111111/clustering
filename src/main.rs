use std::vec;
use crate::cluster_algos::agglomerative::AgglomerativeCluster;
use crate::cluster_algos::dbscan::DBScan;
use crate::cluster_algos::lloyd::Kmeans;
use crate::utils::mathfuncs::{create_square, center_scale};
use cluster_algos::agglomerative::{AggloClusterer, get_partitions};
use plots::scatter_plot;
use ndarray::{array, Array2, Axis};
use std::time::Instant;
pub mod cluster_algos;
pub mod utils;
pub mod plots;

fn main() {
    let cluster_size = 100;
    let noise_intensity = 20;
    let num_clusters = 4;
    let bounds = vec![vec![vec![1.0, 5.0], vec![1.0, 5.0]], vec![vec![1.0, 3.0], vec![5.0, 6.0]], vec![vec![5.0, 6.0], vec![1.0, 3.0]], vec![vec![5.0, 6.0], vec![5.0, 6.0]]];

    let mut data = get_data(noise_intensity, num_clusters, cluster_size, bounds);


    center_scale(&mut data);

    let agglo = true;
    let kmeans = true;
    let dbscan = true;
    let agglo_old = true;

    if dbscan {
        let mut dbscan_model = DBScan::new(&data);
        //model_4.set_epsilon(10e-2);
        //model_4.set_min_points(20);
        let now = Instant::now();
        let partitions_dbscan = dbscan_model.fit_predict(&data);
        println!("DBScan fitted after {:?}", now.elapsed());
        let centroids = array![[0.0, 0.0]];
        let _ = scatter_plot("DBScan_fitted", &data, &partitions_dbscan, &centroids, false);
    }
    if kmeans {
        let mut kmeans_model = Kmeans::new(&data, num_clusters);
        let now = Instant::now();
        let partitions_kmeans = kmeans_model.fit_predict(&data);
        println!("Kmeans fitted after {:?}", now.elapsed());
        let centroids_kmeans = kmeans_model.centroids;
        let _ = scatter_plot("kmeans_fitted", &data, &partitions_kmeans, &centroids_kmeans, true);
    }
    if agglo {
        let mut agglo_model = AggloClusterer::new();
        let now = Instant::now();
        agglo_model.fit(&data);
        println!("Agglo clustereer fitted after {:?}", now.elapsed());
        let partitions_agglo = get_partitions(&agglo_model.retrieve_clusters(num_clusters as usize), &data);
        let centroids = array![[0.0, 0.0]];
        let _ = scatter_plot("AggloScan_fitted", &data, &partitions_agglo, &centroids, false);
    }
    if agglo_old {
        let mut agglo_model_old = AgglomerativeCluster::new(&data, num_clusters as usize);
        let now = Instant::now();
        let partitions_agglo_old = agglo_model_old.fit_predict(&data);
        println!("Old Agglo fitted after {:?}", now.elapsed());
        let centroids = array![[0.0, 0.0]];
        let _ = scatter_plot("AgglomerativeScan_fitted", &data, &partitions_agglo_old, &centroids, false);
    }
    
    println!("all plots generated");
}


fn get_data(noise_intensity: usize, num_clusters: i32, cluster_size: usize, _bounds: Vec<Vec<Vec<f32>>>) -> Array2<f32> {
    //let mut squares: Vec<Array2<ViewRepr<f32>>> = vec![];

    for _i in 0..num_clusters {
        //let x_bound = &bounds[i.to_usize().unwrap()][0];
        //let y_bound = &bounds[i.to_usize().unwrap()][1];
        //let square = create_square(y_bound, x_bound, cluster_size, 2);
        //squares.push(&square.view());
    }

    let square_1: Array2<f32> = create_square(&vec![1.0, 3.0], &vec![2.0, 4.0], cluster_size, 2); // Cluster 1
    let square_2: Array2<f32> = create_square(&vec![5.0, 7.0], &vec![1.0, 3.0], cluster_size, 2); // Cluster 2
    let square_3: Array2<f32> = create_square(&vec![5.0, 7.0], &vec![6.0, 7.0], cluster_size, 2); // Cluster 3
    let square_4: Array2<f32> = create_square(&vec![10.0, 12.0], &vec![6.0, 7.0], cluster_size, 2);
    let square_5: Array2<f32> = create_square(&vec![1.0, 8.0], &vec![1.0, 7.0], cluster_size / 10 + noise_intensity, 2); // A bunch of noise across them all
    	
    /*
    let data: Array2<f32> = ndarray::concatenate(
        Axis(0),
        &squares[0..4],
    )
    .expect("An error occurred while stacking the dataset");
    */

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