use ndarray::prelude::*;
use crate::utils::utility::*;
use crate::utils::mathfuncs::*;
use std::collections::HashSet;


pub struct DBScan {
    pub min_points: usize,
    pub epsilon: f32,
    pub is_visited: HashSet<usize>,
    pub is_in_cluster: HashSet<usize>,
    pub partitions: Vec<i32>,
    pub current_clusters: i32
}

impl DBScan {

    pub fn new(data: &Array2<f32>) -> DBScan {

        DBScan{
            min_points: 10,
            epsilon: 3e-1,
            is_visited: HashSet::new(),
            is_in_cluster: HashSet::new(),
            partitions: vec![0; data.shape()[0]],
            current_clusters: 1
        }
    }

    pub fn set_min_points(&mut self, min_pts: usize) {
        self.min_points = min_pts;
    }

    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon;
    }

    fn initialize(&mut self, data: &Array2<f32>) {
        self.partitions = vec![0; data.shape()[0]];
        self.current_clusters = 1;
    }

    pub fn fit_predict(&mut self, data: &Array2<f32>) -> Vec<i32> {
        self.initialize(data);
        for i in 0..data.shape()[0]{
            if self.is_visited.contains(&i){
                continue;
            }
            self.is_visited.insert(i);
            let neighbours = self.get_neighbours(data, i);
            if neighbours.len() <= self.min_points {
                continue;
            }
            else {
                self.partitions[i] = self.current_clusters;
                self.is_in_cluster.insert(i);
                for point in neighbours.into_iter() {
                    if !self.is_in_cluster.contains(&point){
                        self.expand_cluster(data, point);
                    }
                }
                self.current_clusters += 1;
            }
        }
        return self.partitions.clone();
    }

    fn expand_cluster(&mut self, data: &Array2<f32>,  index: usize) {
        self.partitions[index] = self.current_clusters;
        if !self.is_visited.contains(&index){
            self.is_visited.insert(index);
            self.is_in_cluster.insert(index);
            let neighbours = self.get_neighbours(data, index);
            if neighbours.len() >= self.min_points {
                for point in neighbours.into_iter() {
                    if !self.is_in_cluster.contains(&point){
                        self.expand_cluster(data, point);
                    }
                }
            }
        }

    }

    fn get_neighbours(&self, data: &Array2<f32>, index: usize) -> Vec<usize> {
        let mut neighbours: Vec<usize> = vec![];
        for (i, point) in data.rows().into_iter().enumerate() {
            let dist = l2(&data.row(index).to_owned(), &point.to_owned(), false);
            if dist <= self.epsilon {
                neighbours.push(i);
            }
        }
        return neighbours;
    }

}