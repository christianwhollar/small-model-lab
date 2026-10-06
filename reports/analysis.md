# Small-model experiment

Protocol v2, fixed 150 steps per stage, seeds 17/29/41. These are exploratory synthetic template-shift experiments. The seed-17 pilot remains checked in; it is not pooled with v2.

| Variant | Seed 17 | Seed 29 | Seed 41 | Mean accuracy |
|---|---:|---:|---:|---:|
| lora_teacher | 50.0% | 72.5% | 59.4% | 60.6% |
| full_teacher | 45.6% | 75.0% | 90.6% | 70.4% |
| supervised_student | 75.0% | 65.0% | 59.4% | 66.5% |
| student | 90.0% | 65.0% | 48.8% | 67.9% |
| int8_student | 90.0% | 65.0% | 48.8% | 67.9% |

The distillation result changes materially with the seed. It improves the student in seed 17, ties in seed 29, and harms it in seed 41. The aggregate evidence is too small and unstable to recommend distillation as a default. A validation-selected teacher can still transfer shortcuts that fail on a new template.

In seed 17, int8 weight storage reduces the saved state dictionary from 54,957 to 30,302 bytes (44.9% smaller). Test accuracy is unchanged in these three runs. This implementation dequantizes on every call: seed-17 median forward latency is 0.102 ms for float32 and 0.114 ms for the int8-stored model. Storage compression is the demonstrated benefit; a speedup is not.

The base model is pretrained only to continue ticket text. Its action-token score is not a fair supervised baseline; the supervised student provides that comparison. Softmax confidence is not calibrated, and some confident errors survive the template shift.

The study holds field-order templates out of gradient training, but it uses the same hand-written language and label rules in every split. It does not measure open-ended financial understanding. Seed 17 was visible during development; seeds 29 and 41 were run after the v2 configuration was fixed. Three seeds do not support a precise generalization claim.

Figures can be regenerated with `pip install -e ".[plots]"` and `python scripts/plot_results.py`. All accuracies, sizes, and latencies come from the checked-in JSON.
