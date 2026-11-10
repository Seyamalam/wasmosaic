use std::cell::RefCell;

use wasm_bindgen::prelude::*;

use crate::mat::Mat;

/// Rust-owned vector of matrices used by output families such as `findContours`.
#[wasm_bindgen]
pub struct MatVector {
    values: RefCell<Vec<Mat>>,
}

impl MatVector {
    pub(crate) fn replace(&self, values: Vec<Mat>) {
        self.values.replace(values);
    }

    pub(crate) fn mats(&self) -> Vec<Mat> {
        self.values.borrow().clone()
    }
}

impl Default for MatVector {
    fn default() -> Self {
        Self {
            values: RefCell::new(Vec::new()),
        }
    }
}

#[wasm_bindgen]
impl MatVector {
    /// Returns the number of matrices in the vector.
    #[wasm_bindgen]
    pub fn size(&self) -> usize {
        self.values.borrow().len()
    }

    /// Returns a shared-storage matrix header for one vector element.
    ///
    /// # Errors
    ///
    /// Returns an error when `index` is outside the vector.
    #[wasm_bindgen]
    pub fn get(&self, index: usize) -> Result<Mat, JsError> {
        self.values
            .borrow()
            .get(index)
            .cloned()
            .ok_or_else(|| JsError::new(&format!("MatVector index {index} is out of bounds")))
    }

    /// Appends a shared-storage matrix header.
    #[wasm_bindgen(js_name = push_back)]
    pub fn push_back(&self, value: &Mat) {
        self.values.borrow_mut().push(value.clone());
    }

    /// Removes every vector element.
    #[wasm_bindgen]
    pub fn clear(&self) {
        self.values.borrow_mut().clear();
    }
}

/// Allocates an empty Rust-owned matrix vector.
#[must_use]
#[wasm_bindgen(js_name = matVectorNew)]
pub fn mat_vector_new() -> MatVector {
    MatVector::default()
}
