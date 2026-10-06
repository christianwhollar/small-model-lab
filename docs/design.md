# Controlled experiments on a small language model

The base architecture uses token and position embeddings, pre-norm causal attention, a feed-forward block, and a vocabulary head. The teacher uses width 64 and two blocks; the student uses width 32 and one block. Padding occurs only on the right, so the last real token cannot attend to padding under the causal mask.

Pretraining predicts the next token in the ticket language without supervised action labels. The unadapted base is therefore not expected to solve the downstream action task. Fine-tuning predicts the action token at the last real position. Train/validation/test templates differ in field ordering, which exposes positional shortcuts rather than only memorization of individual record IDs.

LoRA uses frozen base linears plus A and B matrices, with B initialized to zero. The implementation uses a 1/sqrt(rank) scale and leaves adapter computation separate at inference. It demonstrates low-rank adaptation; it does not reproduce every setting or optimized weight merge in the original paper. Tests verify initial functional equivalence, frozen base weights, and nonzero adapter gradients.

Distillation mixes supervised cross-entropy with temperature-scaled KL divergence. Teacher logits are computed without gradients. The supervised student and distilled student begin from identical weights and use the same samples, making the added teacher signal the primary difference. A stronger teacher on validation does not guarantee useful supervision under a new template distribution.

Confidence-threshold coverage is a descriptive measure, not a calibrated guarantee. Forward latency uses warmups and repeated batch-one CPU calls. It excludes model loading, tokenization, HTTP overhead, and autoregressive generation. Small timing differences should not be overinterpreted.

## Walkthrough

Show how a future token is masked, where the supervised label loss is applied, and which weights LoRA changes. Compare validation-selected teacher performance with test-template failures. Explain why the student can outperform its teacher when it also sees true labels. Finally, demonstrate the storage/latency tradeoff of int8 weight storage.

A meaningful extension would train with explicit field-order augmentation, choose settings on a fresh validation protocol, and evaluate once on newly reserved templates or independently authored tickets. The current held-out templates should not be reused as a tuning set while retaining a blind-evaluation claim.
