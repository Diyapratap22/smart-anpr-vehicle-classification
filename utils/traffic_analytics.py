"""Analytics and Utility Functions Module - Traffic Analytics"""
import numpy as np


class TrafficAnalytics:
    """Traffic analytics including density and incident detection."""

    def __init__(self):
        self.vehicle_positions = []
        self.frame_count = 0

    def add_frame_detections(self, detections, frame_shape):
        h, w = frame_shape[:2]
        self.frame_count += 1
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            box_area = (x2 - x1) * (y2 - y1)
            self.vehicle_positions.append({
                'frame': self.frame_count,
                'class': det['class_name'],
                'center_x': center_x,
                'center_y': center_y,
                'area': box_area,
                'confidence': det['confidence']
            })

    def get_vehicle_counts_by_class(self):
        from collections import Counter
        class_counts = Counter()
        for vp in self.vehicle_positions:
            class_counts[vp['class']] += 1
        return dict(class_counts)

    def get_density(self):
        if self.frame_count == 0:
            return 0
        return len(self.vehicle_positions) / self.frame_count

    def detect_incidents(self, threshold_violation_count=5):
        incidents = []
        if len(self.vehicle_positions) < 3:
            return incidents

        vehicle_array = np.array([
            [vp['center_x'], vp['center_y'], vp['area']]
            for vp in self.vehicle_positions
        ])

        if len(vehicle_array) < threshold_violation_count:
            return incidents

        if len(vehicle_array) > 1:
            distances = []
            for i in range(len(vehicle_array)):
                for j in range(i + 1, min(i + 4, len(vehicle_array))):
                    dist = np.sqrt(
                        (vehicle_array[i][0] - vehicle_array[j][0]) ** 2 +
                        (vehicle_array[i][1] - vehicle_array[j][1]) ** 2
                    )
                    distances.append(dist)

            if distances:
                avg_dist = np.mean(distances)
                max_dist = max(distances)

                if avg_dist < 50 and max_dist < 100:
                    incidents.append({
                        'type': 'Potential Congestion',
                        'description': f'High vehicle density detected ({len(self.vehicle_positions)} vehicles), average spacing: {avg_dist:.1f} pixels',
                        'severity': 'warning',
                        'total_vehicles': len(self.vehicle_positions)
                    })

        return incidents