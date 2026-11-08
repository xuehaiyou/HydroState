"""Existing (0,20)% stations: all available dates, conservative static ET support."""

_base_ = "./site_production.py"
min_date_gap_days = 0
class_confidence = 0.50
