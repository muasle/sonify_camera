import time
from user_files.Ellis_FinalProj.camera_bridge import ObjectTracker
from mmm_python import *

mmm_audio = MMMAudio(128, graph_name="Sonify", package_name="user_files.Ellis_FinalProj")
mmm_audio.start_audio()

ping = PolyPal(mmm_audio, "ping", 5)

last_ping = 0.0

def position_update(x, y, confidence, pulse_rate):
    global last_ping
    
    if confidence > 0.5:
        cur_time = time.time()

        interval = 1.0/pulse_rate
        
        if(cur_time - last_ping) >= interval:
            freq = 400 + (pulse_rate * 100)
            ping.send_floats([freq, 0.4])
            last_ping = cur_time

tracker = ObjectTracker(
    url="http://192.168.1.78:8080/video",
    target="book",
    display=True
)

tracker.pos_callback(position_update)

tracker.run()

# Play a single beep
def play_beep(frequency=800.0, volume=0.3):
    ping.send_floats([frequency, volume])

# Test it
play_beep()  # Default 800 Hz beep
play_beep(600.0, 0.5)  # Lower pitch, louder
play_beep(1200.0, 0.2)  # Higher pitch, quieter

mmm_audio.stop_audio()
