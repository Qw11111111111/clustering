use ndarray::prelude::*;


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

pub fn argwhere(x: Vec<i32>, value: i32) -> usize {
    for (i, val) in x.into_iter().enumerate() {
        if value == val{
            return i;
        }
    }
    return usize::MAX;
}