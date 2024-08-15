use ndarray::prelude::*;
use num::ToPrimitive;
use crate::utils::mathfuncs::*;
use std::boxed::Box;
use std::rc::Rc;

pub struct Cluster {
    cluster_1: Option<Rc<Cluster>>,
    cluster_2: Option<Rc<Cluster>>,
    pub members: Vec<usize>,
    index: usize,
    center: Array1<f32>
}

impl Cluster {
    fn new(item: usize, center: Array1<f32>) -> Self {
        Self {
            cluster_1: None, 
            cluster_2: None,
            members: vec![item],
            index: 0,
            center: center
        }
    }

    fn next_cluster(cluster_1: Rc<Cluster>, cluster_2: Rc<Cluster>, index: usize) -> Self {
        let mut members = Vec::new();
        members.append(cluster_1.members.clone().as_mut());
        members.append(cluster_2.members.clone().as_mut());
        let mut center = Array1::zeros(cluster_1.center.len());
        for i in 0..center.len() {
            center[i] = *cluster_1.center.get(i).unwrap() + *cluster_2.center.get(i).unwrap();
        } 
        Self {
            cluster_1: Some(Rc::clone(&cluster_1)),
            cluster_2: Some(Rc::clone(&cluster_2)),
            members: members,
            index: index,
            center: center
        }
    }
}

pub struct AggloClusterer {
    pub head: Option<Rc<Cluster>>,
}

impl AggloClusterer {
    pub fn new() -> Self {
        Self {
            head: None
        }
    }

    fn insert(&mut self, mut cluster: Cluster) {
        if let Some(cur) = &self.head {
            for (i, coord) in cluster.center.iter_mut().enumerate() {
                *coord += *cur.center.get(i).unwrap();
            }
            cluster.members.append(cur.members.clone().as_mut());
            cluster.cluster_2 = Some(Rc::clone(cur));
        }
        self.head = Some(Rc::new(cluster));
    }

    pub fn retrieve_clusters(&self, n_clusters: usize) -> Vec<Rc<Cluster>> {
        //println!("retrieveing");
        if let Some(mut current) = self.head.clone() {
            if n_clusters == 1 {
                return vec![Rc::clone(&current)];
            }
            //let mut clusters = Vec::new();
            let mut all_clusters = Vec::new();
            while all_clusters.len() < n_clusters {
                if all_clusters.len() > 1 {
                    all_clusters.remove(0);
                }
                if let Some(cluster_1) = &current.cluster_1 {
                    all_clusters.push(Rc::clone(cluster_1));
                }
                if let Some(cluster_2) = &current.cluster_2 {
                    all_clusters.push(Rc::clone(cluster_2));
                }
                all_clusters.sort_by(|a, b| {
                    b.index.cmp(&a.index)
                });
                current = Rc::clone(&all_clusters[0]);
            }
            //println!("done");
            all_clusters
        }
        else {
            Vec::new()
        }
    }

    pub fn fit(&mut self, data: &Array2<f32>) {
        let mut all_clusters: Vec<Rc<Cluster>> = data.axis_iter(Axis(0)).enumerate().map(|(i, _item)| {
            Rc::new(Cluster::new(i, data.row(i).clone().to_owned()))
        }).collect();
        let mut index = 1;
        let mut min: f32;
        let mut min_idx: (usize, usize);
        //println!("{}", all_clusters.len());
        while all_clusters.len() > 1 {
            min = f32::MAX;
            min_idx = (0, 0);
            for i in 0..all_clusters.len() - 1 {
                for j in i + 1..all_clusters.len() {
                    let mut center_1 = all_clusters[i].center.clone();
                    let mut center_2 = all_clusters[j].center.clone();
                    center_1 /= all_clusters[i].members.len() as f32;
                    center_2 /= all_clusters[j].members.len() as f32;
                    let distance = l2(&center_1, &center_2, false);
                    if distance < min {
                        min_idx = (i, j);
                        min = distance;
                    }
                }
            }
            let new_cluster = Cluster::next_cluster(Rc::clone(&all_clusters[min_idx.0]), Rc::clone(&all_clusters[min_idx.1]), index);
            index += 1;
            all_clusters[min_idx.0] = Rc::new(new_cluster);
            all_clusters.remove(min_idx.1);
            //println!("{}", all_clusters.len());
        }
        self.head = Some(all_clusters[0].clone());
    }
}

pub fn get_partitions(clusters: &Vec<Rc<Cluster>>, data: &Array2<f32>) -> Vec<i32> {
    //println!("{}", data.len_of(Axis(0)));
    let mut partitions = Vec::new();
    for i in 0..data.len_of(Axis(0)) {
        for (j, cluster) in clusters.iter().enumerate() {
            if cluster.members.contains(&i) {
                partitions.push(j as i32);
            }
        }
    }
    //println!("{}", partitions.len());
    partitions
}

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