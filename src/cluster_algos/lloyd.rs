use std::vec;
use rand::prelude::*;
use ndarray::{Array, Array1, Array2};
use num::ToPrimitive;
use crate::utils::mathfuncs::{silhouette_score, l2, cumsum};

pub struct Kmeans {
    pub centers: i32 ,
    pub max_centers: i32,
    pub accept: f32,
    pub centroids: Array2<f32>,
    pub partition: Vec<i32>,
    pub initializer: &'static str,
    pub max_iter: i32,
    pub retries: i32
}

impl Kmeans {

    pub fn new(data: &Array2<f32>, centers: i32) -> Kmeans {
        Kmeans {
            centers,
            max_centers: 10,
            accept: 0.7,
            centroids: Array::<f32,_>::zeros((centers.to_usize().unwrap(), data.raw_dim()[1])),
            partition: vec![0; data.shape()[0]],
            initializer: "kmeans++",
            max_iter: 100,
            retries: 10
        }
    }

    pub fn config_silhouette(&mut self, accepted_score: f32) {
        self.accept = accepted_score;
    }

    pub fn set_initializer(&mut self, initializer: &'static str) {
        self.initializer = initializer; 
    }

    pub fn set_fitting_time(&mut self, max_iter: i32, retries: i32) {
        self.retries = retries;
        self.max_iter = max_iter;
    }

    pub fn set_max_centers(&mut self, max_centers: i32) {
        self.max_centers = max_centers;
    }

    fn initialize(&mut self, data: &Array2<f32>, n_centers: i32){
        self.centroids = Array::<f32,_>::zeros((n_centers.to_usize().unwrap(), data.raw_dim()[1]));
        if self.initializer == "random_choice" {
            random_choice(data, &mut self.centroids);
        }
        else if self.initializer == "kmeans++" {
            kmeanspp(data, &mut self.centroids);
        } 
        else {
            random_choice(data, &mut self.centroids);
        }
    }

    pub fn fit_predict(&mut self, data: &Array2<f32>) -> Vec<i32> {
        if self.centers > 0 {
            let mut best_centroids = Array2::<f32>::zeros((self.centroids.shape()[0], self.centroids.shape()[1]));
            let mut best_partition: Vec<i32> = vec![0; self.partition.len()];
            let mut minimum = -f32::INFINITY;
            for _ in 0..self.retries{
                self.initialize(data, self.centers);
                let mut count = 0;
                let mut last_centroids = Array2::zeros((self.centroids.shape()[0], self.centroids.shape()[1]));
                while check_if_finished(&self.centroids - last_centroids.clone()) && count < self.max_iter {
                    count += 1;
                    last_centroids = self.centroids.clone();
                    self.update_partitions(data);
                    self.update_centroids(data);
                }
                let score = silhouette_score(data.clone(), self.partition.clone(), self.centroids.clone());
                if score > minimum {
                    minimum = score;
                    best_centroids = self.centroids.clone();
                    best_partition = self.partition.clone();
                }
            }
            self.partition = best_partition;
            self.centroids = best_centroids;
        }
        else {
            let mut best_centroids = Array2::<f32>::zeros((self.centroids.shape()[0], self.centroids.shape()[1]));
            let mut best_partition: Vec<i32> = vec![0; self.partition.len()];
            let mut minimum = -f32::INFINITY;
            for _ in 0..self.retries {
                for i in 2..self.max_centers{
                    self.initialize(data, i);
                    let mut last_centroids = Array2::zeros((self.centroids.shape()[0], self.centroids.shape()[1]));
                    let mut count = 0;

                    while check_if_finished(&self.centroids - last_centroids.clone()) && count < self.max_iter {
                        count += 1;
                        last_centroids = self.centroids.clone();
                        self.update_partitions(data);
                        self.update_centroids(data);
                    }
                    let score = silhouette_score(data.clone(), self.partition.clone(), self.centroids.clone());
                    if score > minimum {
                        minimum = score;
                        best_centroids = self.centroids.clone();
                        best_partition = self.partition.clone();
                    }
                    if score > self.accept{
                        break;
                    }
                }
            }
            self.partition = best_partition;
            self.centroids = best_centroids;
        }
        return self.partition.clone();
    }

    fn update_centroids(&mut self, data: &Array2<f32>) {
        for (i, centroid) in self.centroids.clone().rows_mut().into_iter().enumerate() {
            let mut mean = Array::<f32, _>::zeros(centroid.shape());
            let mut counter = 0.0;
            for (j, num) in self.partition.clone().into_iter().enumerate() {
                if num == i.clone().to_i32().unwrap() {
                    for k in 0..centroid.shape()[0] {
                        mean[k] += data[[j, k]]; 
                    }
                    counter += 1.0;
                }
            }
            if mean == Array::<f32, _>::zeros(centroid.shape()) {
                continue;
            }
            for k in 0..centroid.shape()[0] {
                self.centroids[[i, k]] = mean[k] / counter;
            }
        }
    }

    fn update_partitions(&mut self, data: &Array2<f32>) {
        //let mut rng = thread_rng();
        for (i, point) in data.rows().into_iter().enumerate() {
            let mut min = f32::INFINITY;
            let mut best_centroid: usize = 0;
            /*
            print_array1d(&point.to_owned());
            let mut str = i.to_string();
            println!("{str} i");
            */
            for (j, centroid) in self.centroids.rows().into_iter().enumerate() {
                let dist = l2(&point.to_owned(), &centroid.to_owned(), false);
                if dist <= min {
                    min = dist;
                    best_centroid = j;
                }
            }
            self.partition[i] = best_centroid.to_i32().unwrap();
        }
    }

}

fn random_choice(data: &Array2<f32>, centroids: &mut Array2<f32>) {
    let mut rng = thread_rng();
    let random_vec = vec![rng.gen_range(0..data.shape()[0]); centroids.shape()[0]];
    for i in 0..centroids.shape()[0]{
        for j in 0..centroids.shape()[1]{
            centroids[[i, j]] = data[[random_vec[i], j]];
        }
    }
}

fn kmeanspp(data: &Array2<f32>, centroids: &mut Array2<f32>) {
    let mut rng = thread_rng();
    let mut points = vec![rng.gen_range(0..data.shape()[0])];
    let clone_centroid = centroids.clone();
    replace_values(centroids, &data, 0, points[0]);
    for i in 1..centroids.shape()[0] {
        let mut probs = Array::<f32, _>::zeros(data.shape()[0]);
        for (j, point) in data.rows().into_iter().enumerate() {
            if is_in_vec(&points, &j){
                continue;
            }
            probs[j] = get_smallest_dist(point.to_owned(), clone_centroid.clone());
        }
        probs /= probs.len().to_f32().unwrap();
        cumsum(&mut probs);
        let random_num = rng.gen_range(0.0..1.0);
        for (j, prob) in probs.into_iter().enumerate() {
            if random_num < prob {
                points.push(j);
                replace_values(centroids, data, i, j);
                break;
            }
        }
    }
}

fn check_if_finished(x: Array2<f32>) -> bool {
    for row in x.rows().into_iter() {
        for num in row.into_iter() {
            if *num != 0.0{
                return true;
            }
        }
    }
    return false;
}

fn replace_values (arr1: &mut Array2<f32>, arr2: &Array2<f32>, row: usize, row2: usize) {
    for i in 0..arr1.shape()[1] {
        arr1[[row, i]] = arr2[[row2, i]];
    }
}

fn is_in_vec(vector: &Vec<usize>, value: &usize) -> bool {
    for val in vector.into_iter() {
        if val == value {
            return true;
        }
    }
    return false;
}

fn get_smallest_dist (point: Array1<f32>, data: Array2<f32>) -> f32 {
    let mut minimum = f32::INFINITY;
    let zeros = Array1::zeros(point.shape()[0]);
    for point2 in data.rows().into_iter() {
        if point2 == zeros  {
            continue;
        }
        let dist: f32 = l2(&point, &point2.to_owned(), false);
        if dist < minimum {
            minimum = dist;
        }
    }
    return minimum;
}
