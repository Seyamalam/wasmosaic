# OpenCV.js 4.13.0 browser inventory

This document defines the parity denominator for the stock OpenCV.js 4.13.0 module selection. The machine-readable ledger is [`parity/upstream-inventory.ts`](../parity/upstream-inventory.ts).

## Result

The ledger contains **488 callable operation families**. Reaching 25% requires **122 fully implemented families**. This count is intentionally stricter than counting a class, module, or partially supported algorithm as one completed item.

| Module     | Operation families | 25% checkpoint within module |
| ---------- | -----------------: | ---------------------------: |
| calib3d    |                 15 |                            4 |
| core       |                 58 |                           15 |
| dnn        |                 12 |                            3 |
| features2d |                112 |                           28 |
| imgproc    |                114 |                           29 |
| objdetect  |                103 |                           26 |
| photo      |                 63 |                           16 |
| video      |                 11 |                            3 |
| **Total**  |            **488** |                      **122** |

The per-module checkpoint is `ceil(module count × 0.25)`. Those rounded module checkpoints add to 124, so they are diagnostic only. The project-wide goal uses `ceil(488 × 0.25) = 122`.

The total consists of 205 namespace functions, 265 class methods, and 18 explicitly selected constructors.

## What one family means

An entry represents one callable name owned by one browser-visible surface. C++ overloads for that name are one family. The same method name on two owners is two families because construction, accepted inputs, state, and return behavior can differ. Getters and setters remain separate because either side can be missing or behave incorrectly.

An operation is fully implemented only when the package matches the pinned browser build for:

- every selected overload that is callable from JavaScript;
- accepted matrix depths, channel counts, scalar forms, and optional arguments;
- return values and output-matrix mutation;
- object state and explicit lifetime behavior where applicable;
- documented errors and edge conditions;
- numerical behavior within a published tolerance.

Supporting one color code, interpolation mode, matrix depth, or overload is partial support and does not increment the implemented-family count.

## Method

The inventory was authored on 2026-08-25 from primary OpenCV 4.13.0 material. The official browser binding selection establishes which callable names are in scope. The official module documentation establishes operation ownership and behavioral scope. The ledger uses project-owned identifiers, alphabetical module and member ordering, an original schema, and original scope language. It does not reproduce upstream comments, declaration layout, prose, or source order.

The stock binding selection is the factual baseline, not implementation material. Contributors must continue to follow the [source-independent compatibility policy](COMPATIBILITY_POLICY.md): do not translate OpenCV kernels, generated bindings, tests, configuration structure, comments, lookup tables, or documentation prose into shipping code.

## Primary sources

- [OpenCV.js 4.13.0 stock browser binding selection](https://github.com/opencv/opencv/blob/4.13.0/platforms/js/opencv_js.config.py)
- [OpenCV.js introduction](https://docs.opencv.org/4.13.0/df/d0a/tutorial_js_intro.html)
- [Using OpenCV.js](https://docs.opencv.org/4.13.0/d0/d84/tutorial_js_usage.html)
- [Core functionality](https://docs.opencv.org/4.13.0/d0/de1/group__core.html)
- [Image processing](https://docs.opencv.org/4.13.0/d7/dbd/group__imgproc.html)
- [Object detection](https://docs.opencv.org/4.13.0/d5/d54/group__objdetect.html)
- [Video analysis](https://docs.opencv.org/4.13.0/d7/de9/group__video.html)
- [Deep neural network module](https://docs.opencv.org/4.13.0/d6/d0f/group__dnn.html)
- [2D features framework](https://docs.opencv.org/4.13.0/da/d9b/group__features2d.html)
- [Computational photography](https://docs.opencv.org/4.13.0/d1/d0d/group__photo.html)
- [Camera calibration and 3D reconstruction](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)

## Boundaries and known limitations

- The denominator covers callable entries explicitly selected by the eight stock modules: `core`, `imgproc`, `objdetect`, `video`, `dnn`, `features2d`, `photo`, and `calib3d`.
- Constants, enum values, data-only types, empty selector classes, generated vector wrappers, `Mat` basic-structure methods, and browser helpers such as `imread` and `imshow` are outside this 488-family denominator. They require separate ledgers and do not earn operation-family parity credit here.
- Only constructors explicitly named in the selected callable surface are counted. The ledger does not infer constructors for empty parameter-holder classes.
- The stock selector uses namespace-qualified owner labels such as `dnn_Net` and `aruco_Dictionary`. Those labels are retained as owner facts. A browser runtime can expose a normalized alias, so each fully qualified runtime spelling still needs a generated-build probe before declarations are frozen.
- `overloadCount` and `runtimeArity` stay `null` until a pinned browser probe records the generated JavaScript call forms. There are now 125 audited families, listed in `AUDITED_BINDING_FORMS` in `parity/upstream-inventory.ts`. The resize audit records four accepted forms (three through six arguments) and an arity-zero dispatcher. Doxygen C++ and Python forms alone do not resolve either field.
- The official prebuilt `opencv.js` bundle was downloaded and attempted under Node during the survey, but initialization did not settle within two minutes. No runtime alias or overload facts were inferred from that inconclusive attempt.
- The official documentation occasionally describes APIs not selected for the stock browser build. Such APIs are excluded unless the pinned selector includes their callable name.

This is an engineering inventory, not a legal conclusion. The project's licensing research and release gate still apply.
