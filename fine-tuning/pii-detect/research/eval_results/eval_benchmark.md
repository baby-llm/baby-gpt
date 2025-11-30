| model | overall_acc | precision | recall | avg_latency_ms | n | fp | fn | subjective_review |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | 41.2% | 75.0% | 25.0% | 594.47 | 17 | 1 | 9 | Very low recall with many misses; occasional false positives; unreliable for PII filtering. |
| Qwen/Qwen3-0.6B | 58.8% | 100.0% | 41.7% | 7894.33 | 17 | 0 | 7 | High precision but weak recall; slow in this run and misses many positives. |
| Qwen/Qwen2.5-1.5B-Instruct | 35.3% | 100.0% | 8.3% | 830.50 | 17 | 0 | 11 | Extremely low recall despite perfect precision; not usable without major tuning. |
| Qwen/Qwen3-1.7B | 47.1% | 80.0% | 33.3% | 11109.05 | 17 | 1 | 8 | Mixed precision/recall with many misses; also very slow in this run. |
| Qwen/Qwen2.5-3B-Instruct | 100.0% | 100.0% | 100.0% | 1173.74 | 17 | 0 | 0 | Best of this run; perfect classification on this set with reasonable latency. |
| Qwen/Qwen3-4B-Instruct-2507 | 64.7% | 87.5% | 58.3% | 3069.86 | 17 | 1 | 5 | Better balance but still several misses; moderate latency. |
| Qwen/Qwen2.5-7B-Instruct | 82.4% | 90.9% | 83.3% | 965.94 | 17 | 1 | 2 | Strong performance with few errors and competitive latency among larger models. |

Notes: Metrics computed from `eval_results/result.json` (17 labeled samples) and `eval_results/latency.json` (avg latency per model in ms). Sorted by model size (smallest to largest). overall_acc reflects exact-match on `is_sensitive` vs expected.
