use ndarray::{arr1, s, Array, Array1, Array2, Axis, Shape};
use num::ToPrimitive;
use rand::{thread_rng, Rng};

pub fn silhouette_score(x: Array2<f32>, assignments: Vec<i32>, centroids: Array2<f32>) -> f32{
    let mut scores: Vec<f32> = vec![0.0; x.shape()[0]];
    for (i, point) in x.rows().into_iter().enumerate() {
        let ass: usize = assignments[i].to_usize().unwrap();
        let parent_dist = l2(&point.to_owned(), &centroids.row(ass).to_owned(), false);
        let mut maximum = f32::INFINITY;
        for (j, centroid) in centroids.rows().into_iter().enumerate() {
            if j == ass{
                continue;
            }
            let dist = l2(&point.to_owned(), &centroid.to_owned(), false);
            if dist < maximum {
                maximum = dist;
            }
        }
        maximum = max(vec![maximum, parent_dist]);
        if maximum == 0.0 || maximum == f32::INFINITY || maximum == -f32::INFINITY {
            scores[i] = 0.0;
            continue;
        }
        scores[i] = (maximum - parent_dist) / maximum;
    }
    return 0.0;
}

pub fn l2(x1: &Array1<f32>, x2: &Array1<f32>, grad: bool) -> f32 {
    if grad {
        //TODO: implement
        return 0.0;
    }
    else {
        return 0.5 * squared1d(x1 - x2).sum().sqrt();
    }
}

pub fn squared1d(mut x: Array1<f32>) -> Array1<f32> {
    for i in 0..x.shape()[0] {
        x[i] *= x[i];
    }
    return x;
}

pub fn argwhere(x: Vec<i32>, value: i32) -> usize {
    for (i, val) in x.into_iter().enumerate() {
        if value == val{
            return i;
        }
    }
    return usize::MAX;
}

pub fn max(x: Vec<f32>) -> f32 {
    let mut maximum = - f32::INFINITY;
    for val in x.into_iter() {
        if val > maximum {
            maximum = val;
        }
    }
    return  maximum;
}

pub fn max_int(x: Vec<i32>) -> i32 {
    let mut maximum = - i32::MAX;
    for val in x.into_iter() {
        if val > maximum {
            maximum = val;
        }
    }
    return  maximum;
}

pub fn create_square(min_max: Vec<f32>, n_points: usize, dim: usize) -> Array2<f32> {
    let mut square = Array::<f32, _>::zeros((n_points, dim));
    let mut rng = thread_rng();
    for i in 0..n_points {
        for j in 0..dim {
            square[[i, j]] = rng.gen_range(min_max[0]..min_max[1]);
        }
    }
    return square
}

pub fn print_vec(vector: &Vec<i32>) {
    for val in vector.into_iter() {
        let string = val.to_string();
        print!("{string} ");
    }
}

pub fn print_array(array: &Array2<f32>) {
    for row in array.rows().into_iter() {
        for value in row.into_iter() {
            let value_str = value.to_string();
            print!("{value_str} ");
        }
        print!(" | ");
    }
}

pub fn print_array1d (array : &Array1<f32>) {
    for value in array.into_iter() {
        let value_str = value.to_string();
        print!("{value_str} ");
    }
}

pub fn cumsum (x: &mut Array1<f32>) {
    let mut last = 0.0;
    for (i, val) in x.into_iter().enumerate() {
        *val += last;
        last = *val;
    }
}

pub fn center_scale(data: &mut Array2<f32>){
    let std = data.std_axis(Axis(0), 1.);
    let mean = data.mean_axis(Axis(0)).unwrap();
    for (i, row) in data.rows_mut().into_iter().enumerate(){
        for (j, num) in row.into_iter().enumerate() {
            *num = (*num - mean[j]) / std[j];
        }
    }
}


pub fn square(x: f32) -> f32 {
    return x * x;
}