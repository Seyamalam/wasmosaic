/** Sum of squared pixel differences; the minimum is the best match. */
export const TM_SQDIFF = 0;
/** Squared difference divided by the product of patch and template norms. */
export const TM_SQDIFF_NORMED = 1;
/** Pixel dot product; the maximum is the best match. */
export const TM_CCORR = 2;
/** Dot product divided by the product of patch and template norms. */
export const TM_CCORR_NORMED = 3;
/** Dot product after removing each channel's mean. */
export const TM_CCOEFF = 4;
/** Correlation after channel centering and norm scaling. */
export const TM_CCOEFF_NORMED = 5;

/** OpenCV-compatible template matching method. */
export type TemplateMatchMode = 0 | 1 | 2 | 3 | 4 | 5;
