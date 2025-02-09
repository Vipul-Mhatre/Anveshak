import cv2
import torch
import numpy as np
import pandas as pd
from flask import Flask, render_template, Response, jsonify
from datetime import datetime, timedelta
import time
from transformers import ViTFeatureExtractor, ViTForImageClassification
from ultralytics import YOLO
import torch.nn as nn
import librosa
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from collections import deque



@dataclass
class DetectionConfig:
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_det: int = 300
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class AlertConfig:
    base_threshold: float = 0.7
    adaptation_rate: float = 0.1
    history_window: int = 100
    min_threshold: float = 0.3
    max_threshold: float = 0.9



class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class DigitalTwin:
    
    WEIGHT_THREAT = 0.55
    WEIGHT_BEHAVIOR = 0.30
    WEIGHT_ENVIRONMENT = 0.15

    def __init__(self):
        self.current_state = {}
        self.simulation_queue = queue.Queue()
        self.history = deque(maxlen=1000)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.anomaly_threshold = 2.0
        
    def update_state(self, detection_data: Dict):
        environment_metrics = self._get_environment_metrics()
        current_state = {
            'timestamp': datetime.now(),
            'detections': detection_data,
            'environment_metrics': environment_metrics,
            'risk_assessment': self._assess_risk(detection_data, environment_metrics)
        }
        self.current_state = current_state
        self.history.append(current_state)
        self.simulation_queue.put(current_state)
        self._run_simulations()
        
    def _get_environment_metrics(self) -> Dict:
        lighting = self._estimate_lighting()
        crowd_density = self._calculate_crowd_density()
        motion_intensity = self._analyze_motion_patterns()
        return {
            'lighting_level': lighting,
            'crowd_density': crowd_density,
            'motion_intensity': motion_intensity,
            'environmental_risk': self._calculate_environmental_risk(lighting, crowd_density, motion_intensity)
        }
    
    def _estimate_lighting(self) -> float:
        if not self.current_state.get('detections'):
            return 0.5
        frame = self.current_state['detections'].get('frame', None)
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return np.mean(gray) / 255.0
        return 0.5
        
    def _calculate_crowd_density(self) -> float:
        if not self.current_state.get('detections'):
            return 0.0
        detections = self.current_state['detections']
        frame_area = detections.get('frame_area', 640 * 480)
        total_occupied_area = 0
        for det in detections.get('boxes', []):
            x1, y1, x2, y2 = det[:4]
            area = (x2 - x1) * (y2 - y1)
            total_occupied_area += area
        return min(1.0, total_occupied_area / frame_area)
        
    def _analyze_motion_patterns(self) -> float:
        if len(self.history) < 2:
            return 0.0
        prev_positions = {}
        current_positions = {}
        for det in self.history[-2]['detections'].get('boxes', []):
            obj_id = det[4] if len(det) > 4 else 0
            prev_positions[obj_id] = ((det[0] + det[2]) / 2, (det[1] + det[3]) / 2)
        for det in self.history[-1]['detections'].get('boxes', []):
            obj_id = det[4] if len(det) > 4 else 0
            current_positions[obj_id] = ((det[0] + det[2]) / 2, (det[1] + det[3]) / 2)
        total_displacement = 0
        count = 0
        for obj_id in current_positions:
            if obj_id in prev_positions:
                displacement = np.sqrt((current_positions[obj_id][0] - prev_positions[obj_id][0])**2 +
                                       (current_positions[obj_id][1] - prev_positions[obj_id][1])**2)
                total_displacement += displacement
                count += 1
        return min(1.0, total_displacement / (count * 100)) if count > 0 else 0.0
        
    def _calculate_environmental_risk(self, lighting: float, crowd_density: float, motion_intensity: float) -> float:
        
        
        risk = (1 - lighting) * 0.3 + crowd_density * 0.4 + motion_intensity * 0.3
        return min(1.0, max(0.0, risk))
        
    def _assess_risk(self, detection_data: Dict, environment_metrics: Dict) -> Dict:
        
        threat = self._detect_threat_indicators(detection_data)['risk_score']
        behavior = self._analyze_behavioral_patterns(detection_data)
        environment = environment_metrics['environmental_risk']
        
        
        threat_norm = self._normalize_and_randomize_value(threat)
        behavior_norm = self._normalize_and_randomize_value(behavior)
        environment_norm = self._normalize_and_randomize_value(environment)
        
        
        frame = detection_data.get('frame', None)
        if frame is not None:
            frame_risk = self._calculate_risk_from_frame(frame)
        else:
            frame_risk = 0.5  
        
        
        saw_risk = (self.WEIGHT_THREAT * threat_norm +
                    self.WEIGHT_BEHAVIOR * behavior_norm +
                    self.WEIGHT_ENVIRONMENT * environment_norm)
                    
        
        wpm_risk = (threat_norm ** self.WEIGHT_THREAT *
                    behavior_norm ** self.WEIGHT_BEHAVIOR *
                    environment_norm ** self.WEIGHT_ENVIRONMENT)
                    
        
        combined_existing = max(saw_risk, wpm_risk)
        weight_frame = 0.2
        weight_existing = 0.8
        overall_risk = weight_existing * combined_existing + weight_frame * frame_risk
        overall_risk = min(1.0, max(0.0, overall_risk))
        
        return {
            'overall_risk_score': overall_risk,
            'threat_indicators': self._detect_threat_indicators(detection_data),
            'behavioral_risk': behavior,
            'environmental_risk': environment,
            'risk_level': self._categorize_risk_level(overall_risk)
        }
        
    def _normalize_and_randomize_value(self, value: float) -> float:
        """
        Takes a risk attribute (assumed to be between 0 and 1), samples a random value 
        from a uniform distribution over [0,1] and returns the average as the normalized value.
        """
        random_sample = np.random.uniform(0, 1)
        normalized_value = (value + random_sample) / 2
        return normalized_value

    def _calculate_risk_from_frame(self, frame: np.ndarray) -> float:
        """
        Converts the frame to grayscale, normalizes the pixel values to [0, 1],
        and then computes a risk factor by combining the mean intensity and a random sample.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm_pixels = gray_frame / 255.0
        mean_intensity = np.mean(norm_pixels)
        random_intensity = np.random.choice(norm_pixels.flatten())
        risk_from_frame = mean_intensity * random_intensity
        return risk_from_frame
        
    def _categorize_risk_level(self, risk_score: float) -> str:
        if risk_score < 0.35:
            return 'LOW'
        elif risk_score < 0.65:
            return 'MEDIUM'
        else:
            return 'HIGH'
            
    
    THREAT_COEFF_CONFIDENCE = 0.6
    THREAT_COEFF_SUSPICIOUS = 0.25
    THREAT_COEFF_RESTRICTED = 0.15

    def _detect_threat_indicators(self, detection_data: Dict) -> Dict:
        boxes = detection_data.get('boxes', [])
        classes = detection_data.get('classes', [])
        confidences = detection_data.get('confidences', [])
        total = len(boxes)
        num_suspicious = 0
        sum_conf = 0.0
        num_restricted = 0
        suspicious_classes = {'knife', 'gun', 'suspicious_package', 'person'}
        
        for i, (box, cls) in enumerate(zip(boxes, classes)):
            conf = confidences[i] if i < len(confidences) else 0.0
            if cls in suspicious_classes:
                num_suspicious += 1
                sum_conf += conf
            if self._is_in_restricted_area(box):
                num_restricted += 1
                
        norm_suspicious = num_suspicious / total if total > 0 else 0.0
        norm_restricted = num_restricted / total if total > 0 else 0.0
        avg_conf = sum_conf / num_suspicious if num_suspicious > 0 else 0.0
        
        risk_score = (self.THREAT_COEFF_CONFIDENCE * avg_conf +
                      self.THREAT_COEFF_SUSPICIOUS * norm_suspicious +
                      self.THREAT_COEFF_RESTRICTED * norm_restricted)
        risk_score = min(1.0, max(0.0, risk_score))
        
        return {
            'suspicious_objects': num_suspicious,
            'restricted_area_violations': num_restricted,
            'normalized_suspicious': norm_suspicious,
            'avg_confidence_suspicious': avg_conf,
            'normalized_restricted': norm_restricted,
            'risk_score': risk_score
        }
        
    def _analyze_behavioral_patterns(self, detection_data: Dict) -> float:
        if len(self.history) < 10:
            return 0.0
        patterns = []
        for state in list(self.history)[-10:]:
            detections = state['detections'].get('boxes', [])
            centers = []
            for det in detections:
                x1, y1, x2, y2 = det[:4]
                centers.append([(x1+x2)/2, (y1+y2)/2])
            patterns.append(centers)
        scores = []
        for i in range(len(patterns)-1):
            if patterns[i] and patterns[i+1]:
                dists = cdist(np.array(patterns[i]), np.array(patterns[i+1]))
                scores.append(np.min(dists))
        if scores:
            return min(1.0, max(0.0, np.mean(scores) / 100.0))
        return 0.0
        
    def _is_in_restricted_area(self, box: List[float]) -> bool:
        restricted = [
            {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100},
            {'x1': 500, 'y1': 500, 'x2': 600, 'y2': 600}
        ]
        x1, y1, x2, y2 = box[:4]
        cx, cy = (x1+x2)/2, (y1+y2)/2
        for area in restricted:
            if area['x1'] <= cx <= area['x2'] and area['y1'] <= cy <= area['y2']:
                return True
        return False
        
    def _run_simulations(self):
        while not self.simulation_queue.empty():
            state = self.simulation_queue.get()
            future_states = self._simulate_future_states(state)
            if future_states:
                pred_risk = max(s['risk_assessment']['overall_risk_score'] for s in future_states)
                if pred_risk > self.current_state['risk_assessment']['overall_risk_score']:
                    self.current_state['risk_assessment']['predicted_risk'] = pred_risk
                    
    def _simulate_future_states(self, state: Dict) -> List[Dict]:
        futures = []
        boxes = state['detections'].get('boxes', [])
        for t in range(1, 6):
            futures.append(self._simulate_step(boxes, t))
        return futures
        
    def _simulate_step(self, objects: List[Any], t: int) -> Dict:
        sim_objs = []
        for obj in objects:
            x1, y1, x2, y2 = obj[:4]
            vx, vy = self._estimate_velocity(obj)
            sim_objs.append([x1+vx*t, y1+vy*t, x2+vx*t, y2+vy*t])
        return {
            'timestamp': datetime.now() + timedelta(seconds=t),
            'detections': {'boxes': sim_objs},
            'risk_assessment': self._assess_risk({'boxes': sim_objs}, self._get_environment_metrics())
        }
        
    def _estimate_velocity(self, obj: List[float]) -> Tuple[float, float]:
        if len(self.history) < 2:
            return 0.0, 0.0
        prev_state = self.history[-2]
        prev_objs = prev_state['detections'].get('boxes', [])
        best, mindist = None, float('inf')
        x1, y1, x2, y2 = obj[:4]
        curr_center = ((x1+x2)/2, (y1+y2)/2)
        for prev in prev_objs:
            px1, py1, px2, py2 = prev[:4]
            prev_center = ((px1+px2)/2, (py1+py2)/2)
            dist = np.sqrt((curr_center[0]-prev_center[0])**2 + (curr_center[1]-prev_center[1])**2)
            if dist < mindist:
                mindist, best = dist, prev
        if best is not None:
            bx1, by1, bx2, by2 = best[:4]
            best_center = ((bx1+bx2)/2, (by1+by2)/2)
            return curr_center[0]-best_center[0], curr_center[1]-best_center[1]
        return 0.0, 0.0

class AdaptiveAlertSystem:
    def __init__(self, config: AlertConfig):
        self.config = config
        self.alert_history = []
        self.current_threshold = config.base_threshold
        self.lstm_predictor = LSTMPredictor(input_size=4, hidden_size=64, num_layers=2, output_size=1)
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.alert_levels = {
            'LOW': 0.3,
            'MEDIUM': 0.6,
            'HIGH': 0.8,
            'CRITICAL': 0.9
        }
        
    def update_threshold(self, new_data: Dict):
        if len(self.alert_history) >= self.config.history_window:
            self.alert_history.pop(0)
        self.alert_history.append(new_data)
        features = self._extract_features(new_data)
        predicted_risk = self._predict_future_risk(features)
        self._adjust_threshold(predicted_risk)
        return self._generate_alerts(new_data, predicted_risk)
    
    def _get_features_from_data(self, data: Dict) -> np.ndarray:
        risk = data.get('risk_score', 0.0)
        crowd = data.get('crowd_density', 0.0)
        motion = data.get('motion_intensity', 0.0)
        tod = datetime.now().hour / 24.0
        return np.array([risk, crowd, motion, tod])
    
    def _extract_features(self, data: Dict) -> np.ndarray:
        features = np.array([[data.get('risk_score', 0.0),
                              data.get('crowd_density', 0.0),
                              data.get('motion_intensity', 0.0),
                              datetime.now().hour / 24.0]])
        if len(self.alert_history) > 10:
            hist = np.array([self._get_features_from_data(h) for h in self.alert_history])
            self.feature_scaler.fit(hist)
            features = self.feature_scaler.transform(features)
        return features
        
    def _predict_future_risk(self, features: np.ndarray) -> float:
        tensor = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            pred = self.lstm_predictor(tensor)
        return pred.item()
        
    def _adjust_threshold(self, predicted_risk: float):
        recent = sum(1 for alert in self.alert_history[-10:] if alert.get('severity', 0) > self.current_threshold)
        factor = (predicted_risk - self.current_threshold)
        hist_factor = (recent - 5) / 10
        adjust = (0.6 * factor + 0.4 * hist_factor) * self.config.adaptation_rate
        self.current_threshold = max(self.config.min_threshold, min(self.config.max_threshold, self.current_threshold + adjust))
        
    def _generate_alerts(self, data: Dict, predicted_risk: float) -> List[Dict]:
        alerts = []
        curr_risk = data.get('risk_score', 0.0)
        if curr_risk > self.alert_levels['CRITICAL'] or predicted_risk > self.alert_levels['CRITICAL']:
            alerts.append(self._create_alert('CRITICAL', curr_risk, predicted_risk))
        elif curr_risk > self.alert_levels['HIGH'] or predicted_risk > self.alert_levels['HIGH']:
            alerts.append(self._create_alert('HIGH', curr_risk, predicted_risk))
        elif curr_risk > self.alert_levels['MEDIUM'] or predicted_risk > self.alert_levels['MEDIUM']:
            alerts.append(self._create_alert('MEDIUM', curr_risk, predicted_risk))
        elif curr_risk > self.alert_levels['LOW'] or predicted_risk > self.alert_levels['LOW']:
            alerts.append(self._create_alert('LOW', curr_risk, predicted_risk))
        return alerts
        
    def _create_alert(self, level: str, curr_risk: float, predicted_risk: float) -> Dict:
        return {
            'timestamp': datetime.now(),
            'level': level,
            'current_risk': curr_risk,
            'predicted_risk': predicted_risk,
            'threshold': self.current_threshold,
            'description': self._generate_alert_description(level, curr_risk, predicted_risk)
        }
        
    def _generate_alert_description(self, level: str, curr_risk: float, predicted_risk: float) -> str:
        diff = predicted_risk - curr_risk
        if diff > 0.1:
            trend = "rapidly increasing"
        elif diff > 0:
            trend = "increasing"
        elif diff < -0.1:
            trend = "rapidly decreasing"
        elif diff < 0:
            trend = "decreasing"
        else:
            trend = "stable"
        return f"{level} risk level detected ({curr_risk:.2f}), {trend} trend (predicted: {predicted_risk:.2f})"

class MultiModalFusion:
    def __init__(self):
        self.visual_features = None
        self.audio_features = None
        self.thermal_features = None
        self.feature_weights = {'visual': 0.5, 'audio': 0.3, 'thermal': 0.2}
        self.history = deque(maxlen=100)
    
    def process_visual(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return None
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        edges = cv2.Canny(gray, 100, 200)
        edge_mean = np.mean(edges)
        if hasattr(self, 'prev_gray') and self.prev_gray.shape == gray.shape:
            motion = cv2.absdiff(self.prev_gray, gray)
            motion_feature = np.mean(motion)
        else:
            motion_feature = 0.0
        self.prev_gray = gray.copy()
        features = np.array([mean_val, std_val, edge_mean, motion_feature])
        return features

    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        if audio_data is None:
            return None
        features = []
        mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=22050)[0]
        features.append(np.mean(spectral_centroids))
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=22050)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=22050)
        features.append(tempo[0])
        return np.array(features)
    
    def process_thermal(self, thermal_data: np.ndarray) -> np.ndarray:
        if thermal_data is None:
            return None
        features = []
        features.append(np.mean(thermal_data))
        features.append(np.std(thermal_data))
        gradients = np.gradient(thermal_data)
        features.append(np.mean(np.abs(gradients)))
        threshold = np.mean(thermal_data) + 2 * np.std(thermal_data)
        hot_spots = thermal_data > threshold
        features.append(np.sum(hot_spots) / thermal_data.size)
        return np.array(features)
    
    def fuse_features(self) -> np.ndarray:
        normalized = []
        if self.visual_features is not None:
            norm_vis = self._normalize_features(self.visual_features)
            normalized.append(norm_vis * self.feature_weights['visual'])
        if self.audio_features is not None:
            norm_audio = self._normalize_features(self.audio_features)
            normalized.append(norm_audio * self.feature_weights['audio'])
        if self.thermal_features is not None:
            norm_thermal = self._normalize_features(self.thermal_features)
            normalized.append(norm_thermal * self.feature_weights['thermal'])
        if normalized:
            fused = np.sum(normalized, axis=0)
            if self.history:
                prev = [h.get('fused_features') for h in self.history if 'fused_features' in h]
                if prev:
                    fused = 0.7 * fused + 0.3 * np.mean(prev, axis=0)
            self.history.append({'fused_features': fused})
            return fused
        return None
        
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        min_val = np.min(features)
        max_val = np.max(features)
        if max_val - min_val > 0:
            return (features - min_val) / (max_val - min_val)
        return features

class SurveillanceSystem:
    def __init__(self):
        self.det_config = DetectionConfig()
        self.alert_config = AlertConfig()
        self.yolo_model = YOLO('yolov8x.pt')
        self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.digital_twin = DigitalTwin()
        self.alert_system = AdaptiveAlertSystem(self.alert_config)
        self.multimodal_fusion = MultiModalFusion()
        self.setup_logging()
        self.fps_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('surveillance.log'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        
    def process_frame(self, frame):
        start_time = time.time()
        detection_future = self.executor.submit(self._run_detection, frame)
        vit_future = self.executor.submit(self._run_vit_analysis, frame)
        fusion_future = self.executor.submit(self._run_fusion_analysis, frame)
        results = detection_future.result()
        vit_outputs = vit_future.result()
        fused_features = fusion_future.result()
        
        detection_data = {
            'frame': frame,
            'frame_area': frame.shape[0] * frame.shape[1],
            'boxes': results[0].boxes.data.cpu().numpy(),
            'classes': results[0].boxes.cls.cpu().numpy(),
            'confidences': results[0].boxes.conf.cpu().numpy()
        }
        
        self.digital_twin.update_state({
            'detections': detection_data,
            'vit_features': vit_outputs.logits.detach().cpu().numpy(),
            'fused_features': fused_features
        })
        
        alerts = self.alert_system.update_threshold({
            'risk_score': self.digital_twin.current_state['risk_assessment']['overall_risk_score'],
            'crowd_density': self.digital_twin.current_state['environment_metrics']['crowd_density'],
            'motion_intensity': self.digital_twin.current_state['environment_metrics']['motion_intensity']
        })
        
        processing_time = time.time() - start_time
        self.fps_history.append(1 / processing_time)
        self.accuracy_history.append(self.digital_twin.current_state['risk_assessment']['overall_risk_score'])
        
        return results, vit_outputs, alerts
        
    def _run_detection(self, frame):
        return self.yolo_model(frame, conf=self.det_config.confidence_threshold)
        
    def _run_vit_analysis(self, frame):
        vit_inputs = self.vit_feature_extractor(frame, return_tensors="pt")
        return self.vit_model(**vit_inputs)
        
    def _run_fusion_analysis(self, frame):
        self.multimodal_fusion.visual_features = self.multimodal_fusion.process_visual(frame)
        return self.multimodal_fusion.fuse_features()
        
    def get_processing_fps(self):
        return np.mean(self.fps_history) if self.fps_history else 0
        
    def get_detection_accuracy(self):
        return np.mean(self.accuracy_history) if self.accuracy_history else 0



app = Flask(__name__)
surveillance_system = SurveillanceSystem()

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        desired_fps = 10
        frame_interval = 1.0 / desired_fps
        while True:
            start_time = time.time()
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.resize(frame, (640, 480))
            results, _, _ = surveillance_system.process_frame(frame)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        cap.release()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/analytics')
def get_analytics():
    return jsonify({
        'digital_twin_state': str(surveillance_system.digital_twin.current_state),
        'alert_threshold': surveillance_system.alert_system.current_threshold,
        'system_metrics': {
            'processing_fps': surveillance_system.get_processing_fps(),
            'detection_accuracy': surveillance_system.get_detection_accuracy()
        }
    })

@app.route('/')
def index():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
