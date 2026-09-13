use wasm_bindgen::prelude::*;

use crate::{imgproc_template, mat::Mat};

#[wasm_bindgen(js_name = matMatchTemplateInto)]
pub fn mat_match_template_into(
    image: &Mat,
    template: &Mat,
    result: &Mat,
    method: i32,
) -> Result<(), JsError> {
    imgproc_template::match_template_into(image, template, result, method, None)
        .map_err(JsError::from)
}

#[wasm_bindgen(js_name = matMatchTemplateMaskedInto)]
pub fn mat_match_template_masked_into(
    image: &Mat,
    template: &Mat,
    result: &Mat,
    method: i32,
    mask: &Mat,
) -> Result<(), JsError> {
    imgproc_template::match_template_into(image, template, result, method, Some(mask))
        .map_err(JsError::from)
}
