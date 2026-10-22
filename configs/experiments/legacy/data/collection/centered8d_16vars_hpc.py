"""Collect directly on an HPC2021 IO node into shared scratch."""

_base_ = ["centered8d_16vars.py"]

samples = 20000
output_dir = "/scr/u/xiaoz/HydroState/samples/centered8d_16vars_n20000_v1"
candidate_dir = "/scr/u/xiaoz/HydroState/candidates"
