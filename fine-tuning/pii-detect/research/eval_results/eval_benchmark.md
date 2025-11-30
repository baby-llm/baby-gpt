| model | overall_acc | precision | recall | avg_latency_ms | n | fp | fn | subjective_review |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | 47.1% | 75.0% | 27.3% | 830.66 | 17 | 1 | 8 | Very low recall with many misses; occasional false positives; unreliable for PII filtering. |
| Qwen/Qwen3-0.6B | 47.1% | 66.7% | 36.4% | 9163.53 | 17 | 2 | 7 | Noisy outputs and weak recall; misses many positives and adds some false alarms. |
| Qwen/Qwen2.5-1.5B-Instruct | 41.2% | 100.0% | 9.1% | 1025.07 | 17 | 0 | 10 | Predominantly false negatives; precision ok but coverage too low to use. |
| Qwen/Qwen3-1.7B | 47.1% | 62.5% | 45.5% | 13599.45 | 17 | 3 | 6 | Several outputs fail to match labels and many misses; moderate precision but weak recall and very slow in this run. |
| Qwen/Qwen2.5-3B-Instruct | 82.4% | 83.3% | 90.9% | 1582.76 | 17 | 2 | 1 | Best of this run; high accuracy with limited misses and manageable false positives. |
| Qwen/Qwen3-4B-Instruct-2507 | 52.9% | 71.4% | 45.5% | 3600.39 | 17 | 2 | 6 | Recall issues and some false positives; needs fine-tuning or better prompting. |
| Qwen/Qwen2.5-7B-Instruct | 70.6% | 75.0% | 81.8% | 1139.73 | 17 | 3 | 2 | Moderate accuracy with both FP and FN; below desired production quality. |

Notes: Metrics computed from `eval_results/result.json` (17 labeled samples) and `eval_results/latency.json` (avg latency per model in ms). Sorted by model size (smallest to largest). overall_acc reflects exact-match on `is_sensitive` vs expected.
