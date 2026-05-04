import cv2
import math
from ultralytics import YOLO
import threading
from collections import deque

class VideoCapture:
    def __init__(self, url, width, height):
        # Setup Capture w/ Frame dimensions
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()


    def start(self):
        # Use a thread for better processing
        threading.Thread(target=self.update, daemon=True).start()
        return self
    

    def update(self):
        # Grab the capture information
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret # Return Status
                self.frame = frame # Frame Return
    

    def read(self):
        # Return the Return Status and Frame
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            else:
                return False, None
            

    def stop(self):
        # Kill Capture
        self.stopped = True
        self.cap.release()


class ObjectTracker:
    def __init__(self, url, target, display=True):
        # Define Object Tracker Info
        self.model = YOLO('yolov8n.pt')

        # Main Tracker Inputs
        self.url = url
        self.target = target
        self.display = display

        self.rotate_frame = "portrait" # If Empty, defaults to Landscape

        # Define Capture Dimensions
        self.cap = None
        self.width = 440
        self.height = 600

        # Rotate Frame Dimensions if Portrait
        if self.rotate_frame == "portrait":
            self.frame_width = self.height
            self.frame_height = self.width
        # Otherwise define as regular
        else:
            self.frame_width = self.width
            self.frame_height = self.height

        self.center_x = self.frame_width / 2
        self.center_y = self.frame_height / 2
        self.max_dist = math.hypot(self.center_x, self.center_y)

        # Audio Params
        self.min_pulse_rate = 0.5
        self.max_pulse_rate = 10.0
        self.log_ratio = math.log(self.max_pulse_rate / self.min_pulse_rate)

        # Track Position
        self.pos_history = deque(maxlen=25)
        self.last_pos = None
        self.pos_update = None
        self.frame_count = 0
        

    def pos_callback(self, callback):
        # Callback Position
        self.pos_update = callback
    

    def get_pos_history(self):
        # Get Position History (up to 25 entries)
        return list(self.pos_history)
    

    def calculate_pulse(self, x, y):
        dist = math.hypot(x - self.center_x, y - self.center_y)
        normal_dist = min(dist / self.max_dist, 1.0)
        closeness = 1.0 - normal_dist

        return self.min_pulse_rate * math.exp(self.log_ratio * closeness)


    def process_frame(self, frame):
        if self.rotate_frame == "portrait":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        results = self.model(frame, verbose=False, imgsz=320)
        detections = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if results[0].names[class_id] == self.target:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                x_pos = (x1 + x2) // 2
                y_pos = (y1 + y2) // 2

                self.last_pos = (x_pos, y_pos)
                self.pos_history.append(self.last_pos)

                pulse_rate = self.calculate_pulse(x_pos, y_pos)

                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'x_pos': x_pos, 'y_pos': y_pos,
                    'confidence': confidence,
                    'pulse_rate': pulse_rate
                })

                if self.pos_update:
                    self.pos_update(x_pos, y_pos, confidence, pulse_rate)

        return frame, detections
    

    '''def draw_detections(self, frame, detections):
        # Define a frame to draw detection
        display_frame = frame.copy()

        for i in detections:
            cv2.rectangle(display_frame, 
                          (i['x1'], i['y1']),
                          (i['x2'], i['y2']),
                          self.color, self.label_thickness)
            
            label = f"{i['class_name']}: {i['confidence']:.2f}"
            cv2.putText(display_frame, label, 
                       (i['x1'], i['y1']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, self.label_thickness)
            
            cv2.circle(display_frame, 
                      (i['x_pos'], i['y_pos']), 
                      5, self.color, -1)
            
            position_text = f"X: {i['x_pos']}, Y: {i['y_pos']}"
            cv2.putText(display_frame, position_text, 
                       (i['x1'], i['y2']+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, self.label_thickness)
            
        # Draw trajectory
        if len(self.pos_history) > 1:
            for i in range(1, len(self.pos_history)):
                cv2.line(display_frame, 
                        self.pos_history[i-1], 
                        self.pos_history[i], 
                        (255, 255, 0), self.line_thickness)
            
            for pos in self.pos_history:
                cv2.circle(display_frame, pos, 2, (0, 255, 255), -1)
        
        # Display tracking info
        tracking_info = f"Tracking: {len(self.pos_history)} points"
        cv2.putText(display_frame, tracking_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.last_pos:
            cv2.putText(display_frame, 
                       f"Last: ({self.last_pos[0]}, {self.last_pos[1]})", 
                       (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame'''
    

    def run(self):
        # Run the capture
        self.cap = VideoCapture(self.url, self.width, self.height).start()

        try:
            while True:
                # Grab Frame information
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                self.frame_count += 1

                # Process every other frame to decrease lag
                if self.frame_count % 2 == 0:
                    processed_frame, detections = self.process_frame(frame)
                    
                    # Draw the detections if display is on
                    if self.display:
                        self.render(processed_frame, detections)

        finally:
            self.stop()

    
    def render(self, frame, detections):
        for d in detections:
            x1, y1 = d['x1'], d['y1']
            x2, y2 = d['x2'], d['y2']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Rate: {d['pulse_rate']:.1f}Hz", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        cv2.imshow("Metal Detector View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop()


    def stop(self):
        if hasattr(self, 'cap'): self.cap.stop()
        cv2.destroyAllWindows()
    