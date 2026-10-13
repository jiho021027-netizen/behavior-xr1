# Dataset metadata verified

Only four metadata files were downloaded (74,051 bytes); no video/data shards. `info.json`: 100 tasks, 20,000 episodes, 30 FPS, R1Pro, `observation.state` float32 `[61]`, `action` float32 `[23]`. RGB: left/right RealSense 480x480x3 and ZED 720x720x3. Depth metadata is metric/log encoded with output unit mm. `tasks.jsonl` and `tasks.parquet` each contain 100 task rows. `stats.json` contains state/action/index/timestamp and camera keys. No dedicated skill feature is declared.
