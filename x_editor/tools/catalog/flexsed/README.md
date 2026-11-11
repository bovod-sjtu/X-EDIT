# FlexSED Tool Service

This catalog entry wraps the official `JHU-LCAP/FlexSED` repository as an
X-Editor perception service for open-vocabulary sound event detection.

## Setup

```bash
cd x_editor/tools/catalog/flexsed
./setup.sh
export FLEXSED_REPO_DIR=/path/to/FlexSED
export FLEXSED_DEVICE=cuda
.venv/bin/python service.py --host 127.0.0.1 --port 8057
```

The service does not vendor model weights. The official `api.FlexSED` class
downloads `ckpts/flexsed_as.pt` from Hugging Face when needed unless
`FLEXSED_CKPT_PATH` points at an existing checkpoint.

## Timestamps

FlexSED returns frame probabilities with shape `[num_events, 1, T]`. The
wrapper squeezes this to `[num_events, T]` and thresholds each row:

- frame rate: `25 Hz`
- frame step: `0.04 s`
- `start_sec = frame_start / 25`
- `end_sec = (frame_end + 1) / 25`

Adjacent active regions separated by `merge_gap_sec` or less are merged, and
regions shorter than `min_duration_sec` are dropped.

## Heatmap

The response includes:

- `heatmap.png_path`: visual heatmap, x axis in seconds, y axis in the requested
  event order, color is probability.
- `heatmap.npy_path`: raw score matrix with shape `[num_events, num_frames]`.
- `heatmap.frame_rate_hz`: use this to convert frame indices into seconds.

Use the heatmap as perception evidence for span selection. FlexSED is not ASR
or diarization, so word/speaker scoped edits should still use speech perception
tools.

## Window Top-P Output

For a known time window, pass `start_sec`, `end_sec`, and `p` with a candidate
event list. The service scores each candidate event inside the window and
returns:

```text
[start-end] event1, event2, event3
```

`p` supports two modes:

- `p > 1`: return the top `int(p)` events.
- `0 < p <= 1`: return the smallest sorted event prefix whose normalized
  cumulative window score reaches `p`.

The default `score_mode` is `max`, which is better for short events. Use
`mean` for sustained ambience and `p95` for a robust high-activation score.
