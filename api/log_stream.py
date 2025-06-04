import time
import queue
import threading
import sys

log_queue = queue.Queue()

class StreamToLogger:
    def write(self, message):
        if message.strip():
            log_queue.put(message.strip())
    def flush(self):
        pass

def log_stream():
    while True:
        try:
            message = log_queue.get(timeout=1)
            yield f"data: {message}\n\n"
        except queue.Empty:
            time.sleep(0.1)
