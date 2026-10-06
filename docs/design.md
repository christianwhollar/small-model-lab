# Architecture and decisions

The README describes the current architecture and measured results. This document records the deployment boundary and links decisions to inspectable code.

## Artifact lifecycle

`--download-model` retrieves the versioned release archive and verifies its pinned SHA-256 before extraction. The artifact contains a manifest, safe-tensor weights, tokenizer files, and calibration metadata. Loading verifies component hashes, restricts the model architecture, and loads the tokenizer locally. The service does not deserialize a remote pickle.

The release uses the predeclared seed-17 student. Other seed predictions and reports are retained as research evidence. To use a new trained checkpoint, package its matching tokenizer, metadata, safe tensors and manifest together; do not reuse a confidence threshold from another checkpoint.

The API handles at most 32 nonempty texts per call, each at most 2000 characters, and tokenizes to at most 64 wordpieces. The mean-pooling model loses word order. Long or out-of-domain text can still receive misleading confidence, so the output is a suggestion with a human-review route.

## Reproduce and upgrade

Use the committed constraints and pinned data/model revisions. Published training used MPS and inference timing used CPU. GPU/CPU kernels can vary across platforms. Regenerate test predictions, calibration and latency reports when changing the tokenizer, architecture, precision or runtime.

The eager quantization API is deprecated upstream but works in the tested pinned release. The implementation uses actual quantized kernels and records that they were slower on the measured CPU. Do not infer a speed improvement from file-size reduction alone.

For Docker, download the bundle first into `runtime/banking77-student`, then run `docker compose up --build`. The model is mounted read-only at `/model`; it is not downloaded during image build. Health reports whether a model is loaded. Without an artifact, the study browser is available and prediction returns 503.

## Useful failure demonstrations

- Modify a weight or tokenizer file: integrity verification rejects the artifact.
- Submit empty or oversized text: expect a validation error.
- Inspect confusable intent pairs and the validation-selected review threshold.
- Compare the quantized and float models for both storage and actual CPU latency.
- Compare distillation with its identically initialized supervised control across all three seeds.


## Evolution

The original compact reference implementation is documented in [design-v1.md](design-v1.md). Version 0.2 adds a complete browser workflow, operational state handling, larger experiments or failure studies, and reproducible release artifacts. Earlier studies remain in `reports/`; they have not been replaced with improved numbers under their original names.

See [operations.md](operations.md) for setup and failure demonstrations and [verification.md](verification.md) for the exact validation scope.

## Interface design

Peach panels, coral accents, rounded forms, and a prominent prediction workspace. This model playground uses its own typography, spacing, navigation and component shapes. Fonts and icons are local system fonts and inline SVG, with no external asset requests. The interface supports narrow screens, visible keyboard focus, a skip link, and active navigation semantics.
