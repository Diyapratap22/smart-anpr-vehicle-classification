"""Analytics and Utility Functions Module - Vehicle Counter"""
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, Counter


class VehicleCounter:
    """Tracks vehicle counts and movement across frames."""

    def __init__(self):
        self.counts = defaultdict(int)
        self.frame_counts = defaultdict(int)

    def update(self, detections):
        for det in detections:
            class_name = det['class_name']
            self.counts[class_name] += 1
        return dict(self.counts)

    def get_counts(self):
        return dict(self.counts)

    def get_total(self):
        return sum(self.counts.values())

    def get_breakdown(self):
        return dict(self.counts)