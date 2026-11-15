/** Pinned OpenCV.js color-conversion codes implemented by the first U8 compatibility slice. */
export const COLOR_BGR2BGRA = 0 as const;
export const COLOR_RGB2RGBA = COLOR_BGR2BGRA;
export const COLOR_BGRA2BGR = 1 as const;
export const COLOR_RGBA2RGB = COLOR_BGRA2BGR;
export const COLOR_BGR2RGBA = 2 as const;
export const COLOR_RGB2BGRA = COLOR_BGR2RGBA;
export const COLOR_RGBA2BGR = 3 as const;
export const COLOR_BGRA2RGB = COLOR_RGBA2BGR;
export const COLOR_BGR2RGB = 4 as const;
export const COLOR_RGB2BGR = COLOR_BGR2RGB;
export const COLOR_BGRA2RGBA = 5 as const;
export const COLOR_RGBA2BGRA = COLOR_BGRA2RGBA;
export const COLOR_BGR2GRAY = 6 as const;
export const COLOR_RGB2GRAY = 7 as const;
export const COLOR_GRAY2BGR = 8 as const;
export const COLOR_GRAY2RGB = COLOR_GRAY2BGR;
export const COLOR_GRAY2BGRA = 9 as const;
export const COLOR_GRAY2RGBA = COLOR_GRAY2BGRA;
export const COLOR_BGRA2GRAY = 10 as const;
export const COLOR_RGBA2GRAY = 11 as const;
export const COLOR_BGR2XYZ = 32 as const;
export const COLOR_RGB2XYZ = 33 as const;
export const COLOR_XYZ2BGR = 34 as const;
export const COLOR_XYZ2RGB = 35 as const;
export const COLOR_BGR2YCrCb = 36 as const;
export const COLOR_RGB2YCrCb = 37 as const;
export const COLOR_YCrCb2BGR = 38 as const;
export const COLOR_YCrCb2RGB = 39 as const;
export const COLOR_BGR2YUV = 82 as const;
export const COLOR_RGB2YUV = 83 as const;
export const COLOR_YUV2BGR = 84 as const;
export const COLOR_YUV2RGB = 85 as const;

export type ColorConversionCode =
  | typeof COLOR_BGR2BGRA
  | typeof COLOR_BGRA2BGR
  | typeof COLOR_BGR2RGBA
  | typeof COLOR_RGBA2BGR
  | typeof COLOR_BGR2RGB
  | typeof COLOR_BGRA2RGBA
  | typeof COLOR_BGR2GRAY
  | typeof COLOR_RGB2GRAY
  | typeof COLOR_GRAY2BGR
  | typeof COLOR_GRAY2BGRA
  | typeof COLOR_BGRA2GRAY
  | typeof COLOR_RGBA2GRAY
  | typeof COLOR_BGR2XYZ
  | typeof COLOR_RGB2XYZ
  | typeof COLOR_XYZ2BGR
  | typeof COLOR_XYZ2RGB
  | typeof COLOR_BGR2YCrCb
  | typeof COLOR_RGB2YCrCb
  | typeof COLOR_YCrCb2BGR
  | typeof COLOR_YCrCb2RGB
  | typeof COLOR_BGR2YUV
  | typeof COLOR_RGB2YUV
  | typeof COLOR_YUV2BGR
  | typeof COLOR_YUV2RGB;
