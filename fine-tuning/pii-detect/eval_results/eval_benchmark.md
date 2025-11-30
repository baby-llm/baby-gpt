| model | overall_acc | n | fp | fn | subjective_review |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | 47.1% | 17 | 1 | 8 | Very low recall with many misses; occasional false positives; unreliable for PII filtering. |
| Qwen/Qwen3-0.6B | 47.1% | 17 | 2 | 7 | Noisy outputs and weak recall; misses many positives and adds some false alarms. |
| Qwen/Qwen2.5-1.5B-Instruct | 41.2% | 17 | 0 | 10 | Predominantly false negatives; precision ok but coverage too low to use. |
| Qwen/Qwen3-1.7B | 47.1% | 17 | 0 | 3 | Few false alarms but still low overall accuracy; several answers failed to match expected labels. |
| Qwen/Qwen2.5-3B-Instruct | 82.4% | 17 | 2 | 1 | Best of this run; high accuracy with limited misses and manageable false positives. |
| Qwen/Qwen3-4B-Instruct-2507 | 52.9% | 17 | 2 | 6 | Recall issues and some false positives; needs fine-tuning or better prompting. |
| Qwen/Qwen2.5-7B-Instruct | 70.6% | 17 | 3 | 2 | Moderate accuracy with both FP and FN; below desired production quality. |

Notes: Metrics computed from `eval_results/result.json` (17 labeled samples). Sorted by model size (smallest to largest). overall_acc reflects exact-match on `is_sensitive` vs expected.
