# Operating guide

## Start, stop, and inspect

Start the loopback demo with `python -m smallmodel.serve --demo --download-model` and open port 8105. Stop with Ctrl-C. Persistent data lives under the configured runtime directory. Restarting does not reset edited demo data. Remove or choose a fresh runtime directory only when you deliberately want a fresh synthetic workspace.

For hosted use, configure `API_KEYS_JSON` with strong random keys and server-owned tenant/user/role claims; omit `APP_DEMO`. Terminate TLS, restrict network access and persist application state. Do not publish the built-in demo credentials on an externally reachable service. The included Compose configurations publish to loopback only.

`/health` checks application availability. The three operational services expose authenticated `/metrics` and optional OpenTelemetry export through `OTEL_EXPORTER_OTLP_ENDPOINT`. These metrics do not log raw prompts, API keys, or document bodies. The evaluation and model services expose health and their explicit run/inference results.

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


## Five-minute technical walkthrough

Explain the central data flow in the README, run one successful user workflow, deliberately trigger one failure above, inspect its recorded evidence, and explain one limitation you would address before a broader deployment. Describe measured results as results of this repository's experiments rather than as prior employer production outcomes.
