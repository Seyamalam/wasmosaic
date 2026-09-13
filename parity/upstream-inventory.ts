/**
 * Independently organized inventory of the callable OpenCV.js 4.13.0 browser surface.
 *
 * This file records API facts only. Its schema, identifiers, grouping, ordering, and
 * scope language were authored for this project. It is not generated from, and does
 * not reproduce the structure or prose of, OpenCV's binding configuration.
 */

export const UPSTREAM_MODULES = [
  "calib3d",
  "core",
  "dnn",
  "features2d",
  "imgproc",
  "objdetect",
  "photo",
  "video",
] as const;

export type UpstreamModule = (typeof UPSTREAM_MODULES)[number];
export type OperationKind = "class-method" | "constructor" | "function";

export interface UpstreamOperationFamily {
  /** Stable project-owned identifier, independent of upstream declaration order. */
  readonly id: string;
  /** Unqualified name under `cv`, or `Owner.member` for a class operation. */
  readonly jsNames: readonly string[];
  readonly kind: OperationKind;
  readonly module: UpstreamModule;
  readonly overloadCount: number | null;
  readonly overloadScope: string;
  readonly owner?: string;
  /** JavaScript function length observed in the pinned browser artifact. */
  readonly runtimeArity: number | null;
  readonly scope: string;
  readonly sourceUrls: readonly string[];
}

interface ClassSurface {
  readonly members: readonly string[];
  readonly owner: string;
}

interface ModuleSurface {
  readonly classes: readonly ClassSurface[];
  readonly functions: readonly string[];
  readonly sourceUrl: string;
}

const BROWSER_BASELINE_URL =
  "https://github.com/opencv/opencv/blob/4.13.0/platforms/js/opencv_js.config.py";

interface AuditedBindingForm {
  readonly overloadCount: number;
  readonly runtimeArity: number;
}

const AUDITED_BINDING_FORMS: Readonly<Record<string, AuditedBindingForm>> = Object.freeze({
  "core.function.add-weighted": { overloadCount: 2, runtimeArity: 0 },
  "core.function.bitwise-not": { overloadCount: 2, runtimeArity: 0 },
  "core.function.cart-to-polar": { overloadCount: 2, runtimeArity: 0 },
  "core.function.convert-scale-abs": { overloadCount: 3, runtimeArity: 0 },
  "core.function.count-non-zero": { overloadCount: 1, runtimeArity: 1 },
  "core.function.determinant": { overloadCount: 1, runtimeArity: 1 },
  "core.function.divide": { overloadCount: 3, runtimeArity: 0 },
  "core.function.exp": { overloadCount: 1, runtimeArity: 2 },
  "core.function.flip": { overloadCount: 1, runtimeArity: 3 },
  "core.function.get-optimal-dftsize": { overloadCount: 1, runtimeArity: 1 },
  "core.function.log": { overloadCount: 1, runtimeArity: 2 },
  "core.function.magnitude": { overloadCount: 1, runtimeArity: 3 },
  "core.function.mean": { overloadCount: 2, runtimeArity: 0 },
  "core.function.min-max-loc": { overloadCount: 2, runtimeArity: 0 },
  "core.function.multiply": { overloadCount: 3, runtimeArity: 0 },
  "core.function.polar-to-cart": { overloadCount: 2, runtimeArity: 0 },
  "core.function.pow": { overloadCount: 1, runtimeArity: 3 },
  "core.function.repeat": { overloadCount: 1, runtimeArity: 4 },
  "core.function.rotate": { overloadCount: 1, runtimeArity: 3 },
  "core.function.set-identity": { overloadCount: 2, runtimeArity: 0 },
  "core.function.sqrt": { overloadCount: 1, runtimeArity: 2 },
  "core.function.trace": { overloadCount: 1, runtimeArity: 1 },
  "core.function.transpose": { overloadCount: 1, runtimeArity: 2 },
  "features2d.akaze.get-default-name": { overloadCount: 1, runtimeArity: 0 },
  "features2d.akaze.get-descriptor-channels": { overloadCount: 1, runtimeArity: 0 },
  "features2d.akaze.get-descriptor-size": { overloadCount: 1, runtimeArity: 0 },
  "features2d.akaze.get-descriptor-type": { overloadCount: 1, runtimeArity: 0 },
  "features2d.akaze.get-diffusivity": { overloadCount: 1, runtimeArity: 0 },
  "features2d.akaze.get-noctave-layers": { overloadCount: 1, runtimeArity: 0 },
  "features2d.akaze.get-noctaves": { overloadCount: 1, runtimeArity: 0 },
  "features2d.akaze.get-threshold": { overloadCount: 1, runtimeArity: 0 },
  "features2d.akaze.set-descriptor-channels": { overloadCount: 1, runtimeArity: 1 },
  "features2d.akaze.set-descriptor-size": { overloadCount: 1, runtimeArity: 1 },
  "features2d.akaze.set-descriptor-type": { overloadCount: 1, runtimeArity: 1 },
  "features2d.akaze.set-diffusivity": { overloadCount: 1, runtimeArity: 1 },
  "features2d.akaze.set-noctave-layers": { overloadCount: 1, runtimeArity: 1 },
  "features2d.akaze.set-noctaves": { overloadCount: 1, runtimeArity: 1 },
  "features2d.akaze.set-threshold": { overloadCount: 1, runtimeArity: 1 },
  "features2d.agast-feature-detector.get-default-name": {
    overloadCount: 1,
    runtimeArity: 0,
  },
  "features2d.agast-feature-detector.get-nonmax-suppression": {
    overloadCount: 1,
    runtimeArity: 0,
  },
  "features2d.agast-feature-detector.get-threshold": { overloadCount: 1, runtimeArity: 0 },
  "features2d.agast-feature-detector.get-type": { overloadCount: 1, runtimeArity: 0 },
  "features2d.agast-feature-detector.set-nonmax-suppression": {
    overloadCount: 1,
    runtimeArity: 1,
  },
  "features2d.agast-feature-detector.set-threshold": { overloadCount: 1, runtimeArity: 1 },
  "features2d.agast-feature-detector.set-type": { overloadCount: 1, runtimeArity: 1 },
  "features2d.fast-feature-detector.get-default-name": {
    overloadCount: 1,
    runtimeArity: 0,
  },
  "features2d.fast-feature-detector.get-nonmax-suppression": {
    overloadCount: 1,
    runtimeArity: 0,
  },
  "features2d.fast-feature-detector.get-threshold": { overloadCount: 1, runtimeArity: 0 },
  "features2d.fast-feature-detector.get-type": { overloadCount: 1, runtimeArity: 0 },
  "features2d.fast-feature-detector.set-nonmax-suppression": {
    overloadCount: 1,
    runtimeArity: 1,
  },
  "features2d.fast-feature-detector.set-threshold": { overloadCount: 1, runtimeArity: 1 },
  "features2d.fast-feature-detector.set-type": { overloadCount: 1, runtimeArity: 1 },
  "features2d.gfttdetector.get-block-size": { overloadCount: 1, runtimeArity: 0 },
  "features2d.gfttdetector.get-default-name": { overloadCount: 1, runtimeArity: 0 },
  "features2d.gfttdetector.get-harris-detector": { overloadCount: 1, runtimeArity: 0 },
  "features2d.gfttdetector.get-k": { overloadCount: 1, runtimeArity: 0 },
  "features2d.gfttdetector.get-max-features": { overloadCount: 1, runtimeArity: 0 },
  "features2d.gfttdetector.get-min-distance": { overloadCount: 1, runtimeArity: 0 },
  "features2d.gfttdetector.get-quality-level": { overloadCount: 1, runtimeArity: 0 },
  "features2d.gfttdetector.set-block-size": { overloadCount: 1, runtimeArity: 1 },
  "features2d.gfttdetector.set-harris-detector": { overloadCount: 1, runtimeArity: 1 },
  "features2d.gfttdetector.set-k": { overloadCount: 1, runtimeArity: 1 },
  "features2d.gfttdetector.set-max-features": { overloadCount: 1, runtimeArity: 1 },
  "features2d.gfttdetector.set-min-distance": { overloadCount: 1, runtimeArity: 1 },
  "features2d.gfttdetector.set-quality-level": { overloadCount: 1, runtimeArity: 1 },
  "features2d.kaze.get-default-name": { overloadCount: 1, runtimeArity: 0 },
  "features2d.kaze.get-diffusivity": { overloadCount: 1, runtimeArity: 0 },
  "features2d.kaze.get-extended": { overloadCount: 1, runtimeArity: 0 },
  "features2d.kaze.get-noctave-layers": { overloadCount: 1, runtimeArity: 0 },
  "features2d.kaze.get-noctaves": { overloadCount: 1, runtimeArity: 0 },
  "features2d.kaze.get-threshold": { overloadCount: 1, runtimeArity: 0 },
  "features2d.kaze.get-upright": { overloadCount: 1, runtimeArity: 0 },
  "features2d.kaze.set-extended": { overloadCount: 1, runtimeArity: 1 },
  "features2d.kaze.set-diffusivity": { overloadCount: 1, runtimeArity: 1 },
  "features2d.kaze.set-noctave-layers": { overloadCount: 1, runtimeArity: 1 },
  "features2d.kaze.set-noctaves": { overloadCount: 1, runtimeArity: 1 },
  "features2d.kaze.set-threshold": { overloadCount: 1, runtimeArity: 1 },
  "features2d.kaze.set-upright": { overloadCount: 1, runtimeArity: 1 },
  "features2d.mser.get-default-name": { overloadCount: 1, runtimeArity: 0 },
  "features2d.mser.get-delta": { overloadCount: 1, runtimeArity: 0 },
  "features2d.mser.get-max-area": { overloadCount: 1, runtimeArity: 0 },
  "features2d.mser.get-min-area": { overloadCount: 1, runtimeArity: 0 },
  "features2d.mser.get-pass2-only": { overloadCount: 1, runtimeArity: 0 },
  "features2d.mser.set-delta": { overloadCount: 1, runtimeArity: 1 },
  "features2d.mser.set-max-area": { overloadCount: 1, runtimeArity: 1 },
  "features2d.mser.set-min-area": { overloadCount: 1, runtimeArity: 1 },
  "features2d.mser.set-pass2-only": { overloadCount: 1, runtimeArity: 1 },
  "features2d.orb.get-default-name": { overloadCount: 1, runtimeArity: 0 },
  "features2d.orb.get-fast-threshold": { overloadCount: 1, runtimeArity: 0 },
  "features2d.orb.set-edge-threshold": { overloadCount: 1, runtimeArity: 1 },
  "features2d.orb.set-fast-threshold": { overloadCount: 1, runtimeArity: 1 },
  "features2d.orb.set-first-level": { overloadCount: 1, runtimeArity: 1 },
  "features2d.orb.set-max-features": { overloadCount: 1, runtimeArity: 1 },
  "features2d.orb.set-nlevels": { overloadCount: 1, runtimeArity: 1 },
  "features2d.orb.set-patch-size": { overloadCount: 1, runtimeArity: 1 },
  "features2d.orb.set-scale-factor": { overloadCount: 1, runtimeArity: 1 },
  "features2d.orb.set-score-type": { overloadCount: 1, runtimeArity: 1 },
  "features2d.orb.set-wta-k": { overloadCount: 1, runtimeArity: 1 },
  "imgproc.function.arc-length": { overloadCount: 1, runtimeArity: 2 },
  "imgproc.function.bounding-rect": { overloadCount: 1, runtimeArity: 1 },
  "imgproc.function.contour-area": { overloadCount: 2, runtimeArity: 0 },
  "imgproc.function.get-affine-transform": { overloadCount: 1, runtimeArity: 2 },
  "imgproc.function.create-hanning-window": { overloadCount: 1, runtimeArity: 3 },
  "imgproc.function.invert-affine-transform": { overloadCount: 1, runtimeArity: 2 },
  "imgproc.function.get-rotation-matrix2-d": { overloadCount: 1, runtimeArity: 3 },
  "imgproc.function.get-structuring-element": { overloadCount: 2, runtimeArity: 0 },
  "imgproc.function.is-contour-convex": { overloadCount: 1, runtimeArity: 1 },
  "imgproc.function.point-polygon-test": { overloadCount: 1, runtimeArity: 3 },
  "imgproc.function.resize": { overloadCount: 4, runtimeArity: 0 },
  "photo.tonemap.get-gamma": { overloadCount: 1, runtimeArity: 0 },
  "photo.tonemap.set-gamma": { overloadCount: 1, runtimeArity: 1 },
  "photo.tonemap-drago.get-bias": { overloadCount: 1, runtimeArity: 0 },
  "photo.tonemap-drago.get-saturation": { overloadCount: 1, runtimeArity: 0 },
  "photo.tonemap-drago.set-bias": { overloadCount: 1, runtimeArity: 1 },
  "photo.tonemap-drago.set-saturation": { overloadCount: 1, runtimeArity: 1 },
  "photo.tonemap-mantiuk.get-saturation": { overloadCount: 1, runtimeArity: 0 },
  "photo.tonemap-mantiuk.get-scale": { overloadCount: 1, runtimeArity: 0 },
  "photo.tonemap-mantiuk.set-saturation": { overloadCount: 1, runtimeArity: 1 },
  "photo.tonemap-mantiuk.set-scale": { overloadCount: 1, runtimeArity: 1 },
  "photo.tonemap-reinhard.get-color-adaptation": { overloadCount: 1, runtimeArity: 0 },
  "photo.tonemap-reinhard.get-intensity": { overloadCount: 1, runtimeArity: 0 },
  "photo.tonemap-reinhard.get-light-adaptation": { overloadCount: 1, runtimeArity: 0 },
  "photo.tonemap-reinhard.set-color-adaptation": { overloadCount: 1, runtimeArity: 1 },
  "photo.tonemap-reinhard.set-intensity": { overloadCount: 1, runtimeArity: 1 },
  "photo.tonemap-reinhard.set-light-adaptation": { overloadCount: 1, runtimeArity: 1 },
});

const MODULE_SURFACES = {
  calib3d: {
    classes: [{ owner: "UsacParams", members: ["UsacParams"] }],
    functions: [
      "Rodrigues",
      "calibrateCameraExtended",
      "drawFrameAxes",
      "estimateAffine2D",
      "findHomography",
      "fisheye_initUndistortRectifyMap",
      "fisheye_projectPoints",
      "getDefaultNewCameraMatrix",
      "initUndistortRectifyMap",
      "projectPoints",
      "solvePnP",
      "solvePnPRansac",
      "solvePnPRefineLM",
      "undistort",
    ],
    sourceUrl: "https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html",
  },
  core: {
    classes: [],
    functions: [
      "LUT",
      "absdiff",
      "add",
      "addWeighted",
      "bitwise_and",
      "bitwise_not",
      "bitwise_or",
      "bitwise_xor",
      "cartToPolar",
      "compare",
      "convertScaleAbs",
      "copyMakeBorder",
      "countNonZero",
      "determinant",
      "dft",
      "divide",
      "eigen",
      "exp",
      "flip",
      "gemm",
      "getLogLevel",
      "getOptimalDFTSize",
      "hconcat",
      "inRange",
      "invert",
      "kmeans",
      "log",
      "magnitude",
      "max",
      "mean",
      "meanStdDev",
      "merge",
      "min",
      "minMaxLoc",
      "mixChannels",
      "multiply",
      "norm",
      "normalize",
      "perspectiveTransform",
      "polarToCart",
      "pow",
      "randn",
      "randu",
      "reduce",
      "repeat",
      "rotate",
      "setIdentity",
      "setLogLevel",
      "setRNGSeed",
      "solve",
      "solvePoly",
      "split",
      "sqrt",
      "subtract",
      "trace",
      "transform",
      "transpose",
      "vconcat",
    ],
    sourceUrl: "https://docs.opencv.org/4.13.0/d0/de1/group__core.html",
  },
  dnn: {
    classes: [
      {
        owner: "dnn_Net",
        members: ["forward", "getUnconnectedOutLayersNames", "setInput", "setPreferableBackend"],
      },
    ],
    functions: [
      "blobFromImage",
      "readNet",
      "readNetFromCaffe",
      "readNetFromDarknet",
      "readNetFromONNX",
      "readNetFromTFLite",
      "readNetFromTensorflow",
      "readNetFromTorch",
    ],
    sourceUrl: "https://docs.opencv.org/4.13.0/d6/d0f/group__dnn.html",
  },
  features2d: {
    classes: [
      {
        owner: "AKAZE",
        members: [
          "create",
          "getDefaultName",
          "getDescriptorChannels",
          "getDescriptorSize",
          "getDescriptorType",
          "getDiffusivity",
          "getNOctaveLayers",
          "getNOctaves",
          "getThreshold",
          "setDescriptorChannels",
          "setDescriptorSize",
          "setDescriptorType",
          "setDiffusivity",
          "setNOctaveLayers",
          "setNOctaves",
          "setThreshold",
        ],
      },
      {
        owner: "AgastFeatureDetector",
        members: [
          "create",
          "getDefaultName",
          "getNonmaxSuppression",
          "getThreshold",
          "getType",
          "setNonmaxSuppression",
          "setThreshold",
          "setType",
        ],
      },
      { owner: "BFMatcher", members: ["create", "isMaskSupported"] },
      { owner: "BRISK", members: ["create", "getDefaultName"] },
      {
        owner: "DescriptorMatcher",
        members: [
          "add",
          "clear",
          "clone",
          "create",
          "empty",
          "isMaskSupported",
          "knnMatch",
          "match",
          "radiusMatch",
          "train",
        ],
      },
      {
        owner: "FastFeatureDetector",
        members: [
          "create",
          "getDefaultName",
          "getNonmaxSuppression",
          "getThreshold",
          "getType",
          "setNonmaxSuppression",
          "setThreshold",
          "setType",
        ],
      },
      {
        owner: "Feature2D",
        members: [
          "compute",
          "defaultNorm",
          "descriptorSize",
          "descriptorType",
          "detect",
          "detectAndCompute",
          "empty",
          "getDefaultName",
        ],
      },
      {
        owner: "GFTTDetector",
        members: [
          "create",
          "getBlockSize",
          "getDefaultName",
          "getHarrisDetector",
          "getK",
          "getMaxFeatures",
          "getMinDistance",
          "getQualityLevel",
          "setBlockSize",
          "setHarrisDetector",
          "setK",
          "setMaxFeatures",
          "setMinDistance",
          "setQualityLevel",
        ],
      },
      {
        owner: "KAZE",
        members: [
          "create",
          "getDefaultName",
          "getDiffusivity",
          "getExtended",
          "getNOctaveLayers",
          "getNOctaves",
          "getThreshold",
          "getUpright",
          "setDiffusivity",
          "setExtended",
          "setNOctaveLayers",
          "setNOctaves",
          "setThreshold",
          "setUpright",
        ],
      },
      {
        owner: "MSER",
        members: [
          "create",
          "detectRegions",
          "getDefaultName",
          "getDelta",
          "getMaxArea",
          "getMinArea",
          "getPass2Only",
          "setDelta",
          "setMaxArea",
          "setMinArea",
          "setPass2Only",
        ],
      },
      {
        owner: "ORB",
        members: [
          "create",
          "getDefaultName",
          "getFastThreshold",
          "setEdgeThreshold",
          "setFastThreshold",
          "setFirstLevel",
          "setMaxFeatures",
          "setNLevels",
          "setPatchSize",
          "setScaleFactor",
          "setScoreType",
          "setWTA_K",
        ],
      },
      {
        owner: "SimpleBlobDetector",
        members: ["create", "getDefaultName", "getParams", "setParams"],
      },
    ],
    functions: ["drawKeypoints", "drawMatches", "drawMatchesKnn"],
    sourceUrl: "https://docs.opencv.org/4.13.0/da/d9b/group__features2d.html",
  },
  imgproc: {
    classes: [
      {
        owner: "CLAHE",
        members: [
          "apply",
          "collectGarbage",
          "getClipLimit",
          "getTilesGridSize",
          "setClipLimit",
          "setTilesGridSize",
        ],
      },
      {
        owner: "segmentation_IntelligentScissorsMB",
        members: [
          "IntelligentScissorsMB",
          "applyImage",
          "applyImageFeatures",
          "buildMap",
          "getContour",
          "setEdgeFeatureCannyParameters",
          "setEdgeFeatureZeroCrossingParameters",
          "setGradientMagnitudeMaxLimit",
          "setWeights",
        ],
      },
    ],
    functions: [
      "Canny",
      "GaussianBlur",
      "HoughCircles",
      "HoughLines",
      "HoughLinesP",
      "HuMoments",
      "Laplacian",
      "Scharr",
      "Sobel",
      "adaptiveThreshold",
      "applyColorMap",
      "approxPolyDP",
      "approxPolyN",
      "arcLength",
      "arrowedLine",
      "bilateralFilter",
      "blendLinear",
      "blur",
      "boundingRect",
      "boxFilter",
      "calcBackProject",
      "calcHist",
      "circle",
      "clipLine",
      "compareHist",
      "connectedComponents",
      "connectedComponentsWithStats",
      "contourArea",
      "convertMaps",
      "convexHull",
      "convexityDefects",
      "cornerHarris",
      "cornerMinEigenVal",
      "createCLAHE",
      "createHanningWindow",
      "createLineSegmentDetector",
      "cvtColor",
      "demosaicing",
      "dilate",
      "distanceTransform",
      "distanceTransformWithLabels",
      "divSpectrums",
      "drawContours",
      "drawMarker",
      "ellipse",
      "ellipse2Poly",
      "equalizeHist",
      "erode",
      "fillConvexPoly",
      "fillPoly",
      "filter2D",
      "findContours",
      "findContoursLinkRuns",
      "fitEllipse",
      "fitEllipseAMS",
      "fitEllipseDirect",
      "fitLine",
      "floodFill",
      "getAffineTransform",
      "getFontScaleFromHeight",
      "getPerspectiveTransform",
      "getRectSubPix",
      "getRotationMatrix2D",
      "getStructuringElement",
      "goodFeaturesToTrack",
      "grabCut",
      "integral",
      "integral2",
      "intersectConvexConvex",
      "invertAffineTransform",
      "isContourConvex",
      "line",
      "matchShapes",
      "matchTemplate",
      "medianBlur",
      "minAreaRect",
      "minEnclosingCircle",
      "minEnclosingTriangle",
      "moments",
      "morphologyEx",
      "pointPolygonTest",
      "polylines",
      "preCornerDetect",
      "putText",
      "pyrDown",
      "pyrUp",
      "rectangle",
      "remap",
      "resize",
      "rotatedRectangleIntersection",
      "sepFilter2D",
      "spatialGradient",
      "sqrBoxFilter",
      "stackBlur",
      "threshold",
      "warpAffine",
      "warpPerspective",
      "warpPolar",
      "watershed",
    ],
    sourceUrl: "https://docs.opencv.org/4.13.0/d7/dbd/group__imgproc.html",
  },
  objdetect: {
    classes: [
      {
        owner: "CascadeClassifier",
        members: [
          "CascadeClassifier",
          "detectMultiScale",
          "detectMultiScale2",
          "detectMultiScale3",
          "empty",
          "load",
        ],
      },
      {
        owner: "FaceDetectorYN",
        members: [
          "create",
          "detect",
          "getInputSize",
          "getNMSThreshold",
          "getScoreThreshold",
          "getTopK",
          "setInputSize",
          "setNMSThreshold",
          "setScoreThreshold",
          "setTopK",
        ],
      },
      {
        owner: "GraphicalCodeDetector",
        members: [
          "decode",
          "decodeMulti",
          "detect",
          "detectAndDecode",
          "detectAndDecodeMulti",
          "detectMulti",
        ],
      },
      {
        owner: "HOGDescriptor",
        members: [
          "HOGDescriptor",
          "detectMultiScale",
          "getDaimlerPeopleDetector",
          "getDefaultPeopleDetector",
          "load",
          "setSVMDetector",
        ],
      },
      {
        owner: "QRCodeDetector",
        members: [
          "QRCodeDetector",
          "decode",
          "decodeCurved",
          "decodeMulti",
          "detect",
          "detectAndDecode",
          "detectAndDecodeCurved",
          "detectAndDecodeMulti",
          "detectMulti",
          "setEpsX",
          "setEpsY",
        ],
      },
      {
        owner: "QRCodeDetectorAruco",
        members: [
          "QRCodeDetectorAruco",
          "decode",
          "decodeMulti",
          "detect",
          "detectAndDecode",
          "detectAndDecodeMulti",
          "detectMulti",
          "setArucoParameters",
          "setDetectorParameters",
        ],
      },
      { owner: "QRCodeDetectorAruco_Params", members: ["Params"] },
      {
        owner: "aruco_ArucoDetector",
        members: [
          "ArucoDetector",
          "detectMarkers",
          "refineDetectedMarkers",
          "setDetectorParameters",
          "setDictionary",
          "setRefineParameters",
        ],
      },
      {
        owner: "aruco_Board",
        members: ["Board", "generateImage", "matchImagePoints"],
      },
      {
        owner: "aruco_CharucoBoard",
        members: [
          "CharucoBoard",
          "checkCharucoCornersCollinear",
          "generateImage",
          "getChessboardCorners",
          "getLegacyPattern",
          "getNearestMarkerCorners",
          "matchImagePoints",
          "setLegacyPattern",
        ],
      },
      { owner: "aruco_CharucoParameters", members: ["CharucoParameters"] },
      {
        owner: "aruco_CharucoDetector",
        members: [
          "CharucoDetector",
          "detectBoard",
          "detectDiamonds",
          "setBoard",
          "setCharucoParameters",
          "setDetectorParameters",
          "setRefineParameters",
        ],
      },
      { owner: "aruco_DetectorParameters", members: ["DetectorParameters"] },
      {
        owner: "aruco_Dictionary",
        members: [
          "Dictionary",
          "generateImageMarker",
          "getBitsFromByteList",
          "getByteListFromBits",
          "getDistanceToId",
        ],
      },
      {
        owner: "aruco_GridBoard",
        members: [
          "GridBoard",
          "generateImage",
          "getGridSize",
          "getMarkerLength",
          "getMarkerSeparation",
          "matchImagePoints",
        ],
      },
      { owner: "aruco_RefineParameters", members: ["RefineParameters"] },
      {
        owner: "barcode_BarcodeDetector",
        members: [
          "BarcodeDetector",
          "decode",
          "decodeMulti",
          "decodeWithType",
          "detect",
          "detectAndDecode",
          "detectAndDecodeMulti",
          "detectAndDecodeWithType",
          "detectMulti",
        ],
      },
    ],
    functions: [
      "drawDetectedCornersCharuco",
      "drawDetectedDiamonds",
      "drawDetectedMarkers",
      "extendDictionary",
      "generateImageMarker",
      "getPredefinedDictionary",
      "groupRectangles",
    ],
    sourceUrl: "https://docs.opencv.org/4.13.0/d5/d54/group__objdetect.html",
  },
  photo: {
    classes: [
      { owner: "AlignExposures", members: ["process"] },
      {
        owner: "AlignMTB",
        members: [
          "calculateShift",
          "computeBitmaps",
          "getCut",
          "getExcludeRange",
          "getMaxBits",
          "setCut",
          "setExcludeRange",
          "setMaxBits",
          "shiftMat",
        ],
      },
      { owner: "CalibrateCRF", members: ["process"] },
      {
        owner: "CalibrateDebevec",
        members: ["getLambda", "getRandom", "getSamples", "setLambda", "setRandom", "setSamples"],
      },
      {
        owner: "CalibrateRobertson",
        members: ["getMaxIter", "getRadiance", "getThreshold", "setMaxIter", "setThreshold"],
      },
      { owner: "MergeDebevec", members: ["process"] },
      { owner: "MergeExposures", members: ["process"] },
      {
        owner: "MergeMertens",
        members: [
          "getContrastWeight",
          "getExposureWeight",
          "getSaturationWeight",
          "process",
          "setContrastWeight",
          "setExposureWeight",
          "setSaturationWeight",
        ],
      },
      { owner: "MergeRobertson", members: ["process"] },
      { owner: "Tonemap", members: ["getGamma", "process", "setGamma"] },
      {
        owner: "TonemapDrago",
        members: [
          "getBias",
          "getSaturation",
          "getSigmaColor",
          "getSigmaSpace",
          "setBias",
          "setSaturation",
          "setSigmaColor",
          "setSigmaSpace",
        ],
      },
      {
        owner: "TonemapMantiuk",
        members: ["getSaturation", "getScale", "setSaturation", "setScale"],
      },
      {
        owner: "TonemapReinhard",
        members: [
          "getColorAdaptation",
          "getIntensity",
          "getLightAdaptation",
          "setColorAdaptation",
          "setIntensity",
          "setLightAdaptation",
        ],
      },
    ],
    functions: [
      "createAlignMTB",
      "createCalibrateDebevec",
      "createCalibrateRobertson",
      "createMergeDebevec",
      "createMergeMertens",
      "createMergeRobertson",
      "createTonemapDrago",
      "createTonemapMantiuk",
      "createTonemapReinhard",
      "inpaint",
    ],
    sourceUrl: "https://docs.opencv.org/4.13.0/d1/d0d/group__photo.html",
  },
  video: {
    classes: [
      {
        owner: "BackgroundSubtractor",
        members: ["apply", "getBackgroundImage"],
      },
      {
        owner: "BackgroundSubtractorMOG2",
        members: ["BackgroundSubtractorMOG2", "apply"],
      },
      { owner: "TrackerMIL", members: ["create"] },
    ],
    functions: [
      "CamShift",
      "calcOpticalFlowFarneback",
      "calcOpticalFlowPyrLK",
      "createBackgroundSubtractorMOG2",
      "findTransformECC",
      "meanShift",
    ],
    sourceUrl: "https://docs.opencv.org/4.13.0/d7/de9/group__video.html",
  },
} as const satisfies Record<UpstreamModule, ModuleSurface>;

function stableSegment(value: string): string {
  return value
    .replaceAll(/([a-z0-9])([A-Z])/g, "$1-$2")
    .replaceAll("_", "-")
    .toLowerCase();
}

function constructorLeaf(owner: string): string {
  const segments = owner.split("_");
  return segments[segments.length - 1] ?? owner;
}

function bindingForm(id: string): AuditedBindingForm | undefined {
  return AUDITED_BINDING_FORMS[id];
}

function functionFamily(
  module: UpstreamModule,
  name: string,
  sourceUrl: string,
): UpstreamOperationFamily {
  const id = `${module}.function.${stableSegment(name)}`;
  const audited = bindingForm(id);
  return {
    id,
    jsNames: [name],
    kind: "function",
    module,
    overloadCount: audited?.overloadCount ?? null,
    overloadScope: audited
      ? "The pinned stock browser build exposes one audited generated call form."
      : "All overloads selected by the pinned stock browser build; the generated JS overload count still needs runtime verification.",
    runtimeArity: audited?.runtimeArity ?? null,
    scope: `Match cv.${name} inputs, defaults, output mutation, return values, and documented failure behavior.`,
    sourceUrls: [sourceUrl, BROWSER_BASELINE_URL],
  };
}

function classFamily(
  module: UpstreamModule,
  owner: string,
  member: string,
  sourceUrl: string,
): UpstreamOperationFamily {
  const isConstructor = member === constructorLeaf(owner);
  const id = `${module}.${stableSegment(owner)}.${stableSegment(member)}`;
  const audited = bindingForm(id);
  const common = {
    id,
    jsNames: [`${owner}.${member}`],
    module,
    overloadCount: audited?.overloadCount ?? null,
    overloadScope: audited
      ? "The pinned stock browser build exposes one audited generated call form."
      : "All overloads selected by the pinned stock browser build; the generated JS overload count still needs runtime verification.",
    owner,
    runtimeArity: audited?.runtimeArity ?? null,
    sourceUrls: [sourceUrl, BROWSER_BASELINE_URL],
  } as const;

  if (isConstructor) {
    return {
      ...common,
      kind: "constructor",
      scope: `Create ${owner} instances with the selected arguments and browser-visible lifetime behavior.`,
    };
  }

  return {
    ...common,
    kind: "class-method",
    scope: `Match ${owner}.${member} inputs, defaults, state changes, outputs, and documented failure behavior.`,
  };
}

function buildInventory(): readonly UpstreamOperationFamily[] {
  const families: UpstreamOperationFamily[] = [];

  for (const module of UPSTREAM_MODULES) {
    const surface = MODULE_SURFACES[module];
    for (const name of surface.functions) {
      families.push(functionFamily(module, name, surface.sourceUrl));
    }
    for (const classSurface of surface.classes) {
      for (const member of classSurface.members) {
        families.push(classFamily(module, classSurface.owner, member, surface.sourceUrl));
      }
    }
  }

  return families;
}

export const UPSTREAM_BROWSER_INVENTORY = buildInventory();

function countModule(module: UpstreamModule): number {
  return UPSTREAM_BROWSER_INVENTORY.filter((family) => family.module === module).length;
}

export const UPSTREAM_COUNTS_BY_MODULE = Object.freeze({
  calib3d: countModule("calib3d"),
  core: countModule("core"),
  dnn: countModule("dnn"),
  features2d: countModule("features2d"),
  imgproc: countModule("imgproc"),
  objdetect: countModule("objdetect"),
  photo: countModule("photo"),
  video: countModule("video"),
}) satisfies Readonly<Record<UpstreamModule, number>>;

export const UPSTREAM_OPERATION_FAMILY_COUNT = UPSTREAM_BROWSER_INVENTORY.length;
