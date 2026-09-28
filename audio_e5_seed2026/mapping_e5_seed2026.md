# E5 合奏音频听测索引（len256 / seed2026 / 10 窗口 / 2 模型）

- 渲染管线：V4 原逻辑 `derive_ensemble_v4.py --late-join 0` → `ts_render_full.py --seed 2024 --top_k 10 --win 5.0` → `mix_ensemble.py`
- 声部与音量配比（V4 写死，未改动）：pipa 1.0 / dongxiao 0.9 / sanxian 0.7 / erxian 0.8
- 混音文件目录：`audio_e5_seed2026/mix/`，分轨目录：`audio_e5_seed2026/tracks/<stem>/`

| model | seed | target_len | song | window | source_midi | mix_wav | tracks_done | status |
|---|---|---|---|---|---|---|---|---|
| boyatcn | 2026 | 256 | dapu14kouhuangtian | w0 | dapu14kouhuangtian_w0_boyatcn_seed2026_len256.mid | dapu14kouhuangtian_w0_boyatcn_seed2026_len256_ensemble.wav | 4/4 | OK |
| boyatcn | 2026 | 256 | dapu14kouhuangtian | w1 | dapu14kouhuangtian_w1_boyatcn_seed2026_len256.mid | dapu14kouhuangtian_w1_boyatcn_seed2026_len256_ensemble.wav | 4/4 | OK |
| boyatcn | 2026 | 256 | dapu14kouhuangtian | w2 | dapu14kouhuangtian_w2_boyatcn_seed2026_len256.mid | dapu14kouhuangtian_w2_boyatcn_seed2026_len256_ensemble.wav | 4/4 | OK |
| boyatcn | 2026 | 256 | dapu14kouhuangtian | w3 | dapu14kouhuangtian_w3_boyatcn_seed2026_len256.mid | dapu14kouhuangtian_w3_boyatcn_seed2026_len256_ensemble.wav | 4/4 | OK |
| boyatcn | 2026 | 256 | dapu15wujinjiao | w0 | dapu15wujinjiao_w0_boyatcn_seed2026_len256.mid | dapu15wujinjiao_w0_boyatcn_seed2026_len256_ensemble.wav | 4/4 | OK |
| boyatcn | 2026 | 256 | dapu15wujinjiao | w1 | dapu15wujinjiao_w1_boyatcn_seed2026_len256.mid | dapu15wujinjiao_w1_boyatcn_seed2026_len256_ensemble.wav | 4/4 | OK |
| boyatcn | 2026 | 256 | dapu15wujinjiao | w2 | dapu15wujinjiao_w2_boyatcn_seed2026_len256.mid | dapu15wujinjiao_w2_boyatcn_seed2026_len256_ensemble.wav | 4/4 | OK |
| boyatcn | 2026 | 256 | dapu15wujinjiao | w3 | dapu15wujinjiao_w3_boyatcn_seed2026_len256.mid | dapu15wujinjiao_w3_boyatcn_seed2026_len256_ensemble.wav | 4/4 | OK |
| boyatcn | 2026 | 256 | dapu15wujinjiao | w4 | dapu15wujinjiao_w4_boyatcn_seed2026_len256.mid | dapu15wujinjiao_w4_boyatcn_seed2026_len256_ensemble.wav | 4/4 | OK |
| boyatcn | 2026 | 256 | dapu15wujinjiao | w5 | dapu15wujinjiao_w5_boyatcn_seed2026_len256.mid | dapu15wujinjiao_w5_boyatcn_seed2026_len256_ensemble.wav | 4/4 | OK |
| musicmamba | 2026 | 256 | dapu14kouhuangtian | w0 | dapu14kouhuangtian_w0_musicmamba_seed2026_len256.mid | dapu14kouhuangtian_w0_musicmamba_seed2026_len256_ensemble.wav | 4/4 | OK |
| musicmamba | 2026 | 256 | dapu14kouhuangtian | w1 | dapu14kouhuangtian_w1_musicmamba_seed2026_len256.mid | dapu14kouhuangtian_w1_musicmamba_seed2026_len256_ensemble.wav | 4/4 | OK |
| musicmamba | 2026 | 256 | dapu14kouhuangtian | w2 | dapu14kouhuangtian_w2_musicmamba_seed2026_len256.mid | dapu14kouhuangtian_w2_musicmamba_seed2026_len256_ensemble.wav | 4/4 | OK |
| musicmamba | 2026 | 256 | dapu14kouhuangtian | w3 | dapu14kouhuangtian_w3_musicmamba_seed2026_len256.mid | dapu14kouhuangtian_w3_musicmamba_seed2026_len256_ensemble.wav | 4/4 | OK |
| musicmamba | 2026 | 256 | dapu15wujinjiao | w0 | dapu15wujinjiao_w0_musicmamba_seed2026_len256.mid | dapu15wujinjiao_w0_musicmamba_seed2026_len256_ensemble.wav | 4/4 | OK |
| musicmamba | 2026 | 256 | dapu15wujinjiao | w1 | dapu15wujinjiao_w1_musicmamba_seed2026_len256.mid | dapu15wujinjiao_w1_musicmamba_seed2026_len256_ensemble.wav | 4/4 | OK |
| musicmamba | 2026 | 256 | dapu15wujinjiao | w2 | dapu15wujinjiao_w2_musicmamba_seed2026_len256.mid | dapu15wujinjiao_w2_musicmamba_seed2026_len256_ensemble.wav | 4/4 | OK |
| musicmamba | 2026 | 256 | dapu15wujinjiao | w3 | dapu15wujinjiao_w3_musicmamba_seed2026_len256.mid | dapu15wujinjiao_w3_musicmamba_seed2026_len256_ensemble.wav | 4/4 | OK |
| musicmamba | 2026 | 256 | dapu15wujinjiao | w4 | dapu15wujinjiao_w4_musicmamba_seed2026_len256.mid | dapu15wujinjiao_w4_musicmamba_seed2026_len256_ensemble.wav | 4/4 | OK |
| musicmamba | 2026 | 256 | dapu15wujinjiao | w5 | dapu15wujinjiao_w5_musicmamba_seed2026_len256.mid | dapu15wujinjiao_w5_musicmamba_seed2026_len256_ensemble.wav | 4/4 | OK |
