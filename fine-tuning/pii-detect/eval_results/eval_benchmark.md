| model | overall_acc | easy_acc | medium_acc | hard_acc | n | insight |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-3B-Instruct | 76.0% | 60.0% | 87.5% | 75.0% | 25 | misses positives (FN-heavy) on harder samples; drops on hard set; medium stronger |
| Qwen/Qwen3-4B-Instruct-2507 | 76.0% | 60.0% | 75.0% | 83.3% | 25 | misses positives (FN-heavy) on harder samples; handles hard similar/better than medium |
| Qwen/Qwen2.5-7B-Instruct | 68.0% | 60.0% | 87.5% | 58.3% | 25 | misses positives (FN-heavy) on harder samples; drops on hard set; medium stronger |
| Qwen/Qwen2.5-1.5B-Instruct | 56.0% | 60.0% | 50.0% | 58.3% | 25 | misses positives (FN-heavy) on harder samples; handles hard similar/better than medium |
| Qwen/Qwen3-0.6B | 52.0% | 40.0% | 62.5% | 50.0% | 25 | shows thinking tokens (25/25) but reasoning often noisy; misses positives (FN-heavy) on harder samples; drops on hard set; medium stronger |
| Qwen/Qwen3-1.7B | 36.0% | 40.0% | 12.5% | 50.0% | 25 | shows thinking tokens (18/25) but reasoning often noisy; balanced FP/FN; handles hard similar/better than medium |
| Qwen/Qwen2.5-0.5B-Instruct | 28.0% | 40.0% | 12.5% | 33.3% | 25 | misses positives (FN-heavy) on harder samples; handles hard similar/better than medium |