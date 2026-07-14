# Vendored Basis Universal transcoder

`basis_transcoder.cjs` + `basis_transcoder.wasm` are copied verbatim from
three.js (`examples/jsm/libs/basis/`, three@0.163.0) — the same transcoder
its `KTX2Loader` uses.

`deoptimize.mjs` uses it to decode the asset library's KTX2/Basis (ETC1S/UASTC)
base-color textures to RGBA so they can be re-embedded as PNG (via sharp) that
trimesh reads — giving Stage 2 real per-texel albedo for library cells.

The `.js` is renamed to `.cjs` because this package is `"type": "module"`, and
the transcoder is a CommonJS/UMD module (`module.exports = BASIS`).
