use ndarray::prelude::*;
use num::ToPrimitive;
use crate::utils::mathfuncs::*;
use crate::utils::utility::*;


pub struct AgglomerativeCluster {
    pub centers: usize,
    pub clusters: Vec<Vec<Array1<f32>>>
}

impl AgglomerativeCluster {

    pub fn new(data: &Array2<f32>, centers: usize) -> AgglomerativeCluster {
        
        let mut clusters = vec![vec![data.row(0).to_owned()]];
        for i in 1..data.shape()[0]{
            clusters.append(&mut vec![vec![data.row(i).to_owned()]]);
        }
        AgglomerativeCluster {
            centers,
            clusters: clusters
        }
    }

    fn initialize(&mut self, data: &Array2<f32>) {
        self.clusters = vec![vec![data.row(0).to_owned()]];
        for i in 1..data.shape()[0]{
            self.clusters.append(&mut vec![vec![data.row(i).to_owned()]]);
        }
    }

    pub fn fit_predict(&mut self, data: &Array2<f32>) -> Vec<i32> {
        self.initialize(data);
        while self.clusters.len() > self.centers {
            let best: &Vec<usize> = &self.update();
            let mut vector = self.clusters[best[1]].clone();
            self.clusters[best[0]].append(&mut vector); 
            self.clusters.remove(best[1]);
        }
        let partitions = self.get_partition(data);
        return partitions;
    }

    fn update(&self) -> Vec<usize> {
        let mut minimum = f32::INFINITY;
        let mut best: Vec<usize> = vec![0, 0];
        for i in 0..self.clusters.len() {
            for j in i.. self.clusters.len(){
                if i == j {
                    continue;
                }
                let cluster_1 = &self.clusters[i];
                let cluster_2 = &self.clusters[j];
                let dist = l2(&mean_of_vec_arr(cluster_1), &mean_of_vec_arr(cluster_2), false);
                if dist < minimum {
                    minimum = dist;
                    best[0] = i;
                    best[1] = j;
                }
            }
        }
        best
    }

    fn get_partition(&self, data: &Array2<f32>) -> Vec<i32> {
        let mut partitions = vec![0; data.shape()[0]];
        for (i, center) in self.clusters.clone().into_iter().enumerate() {
            for point in center.into_iter() {
                for (j, datapoint) in data.rows().into_iter().enumerate() {
                    if point == datapoint{
                        partitions[j] = i.to_i32().unwrap();
                    }
                }
            }
        }  

        return partitions;
    }
}