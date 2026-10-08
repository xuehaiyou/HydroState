# SM experiment configurations

Generate fully expanded, immutable configs with
`tools/experiments/prepare_sm_causal.py`. They are written alongside their
manifests, training-only statistics and saved batch schedules under
`outputs/sm_causal_DATE/{screen,full}/`.

The source training config is the frozen actual v2 run, not a moving default.
See `tools/experiments/README.md` for launch commands and staged execution.
