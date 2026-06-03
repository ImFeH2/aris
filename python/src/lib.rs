use pyo3::prelude::*;

#[pymodule]
mod _aris {
    use pyo3::prelude::*;

    #[pyfunction]
    fn add(a: f64, b: f64) -> f64 {
        aris::add(a, b)
    }
}
