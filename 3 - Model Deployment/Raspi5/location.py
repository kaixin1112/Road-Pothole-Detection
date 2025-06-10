import serial
import pynmea2
import statistics
from collections import deque
import threading

class GPSLocator:
    def __init__(self, port='/dev/ttyACM0', baudrate=9600, timeout=1, buffer_size=5, min_sats=6):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.lat_buffer = deque(maxlen=buffer_size)
        self.lon_buffer = deque(maxlen=buffer_size)
        self.min_sats = min_sats
        self.latest_location = None
        self.lock = threading.Lock()
        self.running = True

        # Start background thread
        self._thread = threading.Thread(target=self._gps_loop, daemon=True)
        self._thread.start()

    def _gps_loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode('ascii', errors='replace').strip()
                if line.startswith('$GPGGA'):
                    msg = pynmea2.parse(line)

                    if int(msg.gps_qual) >= 1 and int(msg.num_sats) >= self.min_sats:
                        lat = msg.latitude
                        lon = msg.longitude
                        smoothed_lat, smoothed_lon = self.smooth_coordinates(lat, lon)

                        with self.lock:
                            self.latest_location = {
                                'latitude': smoothed_lat,
                                'longitude': smoothed_lon,
                                'raw_latitude': lat,
                                'raw_longitude': lon,
                                'satellites': int(msg.num_sats),
                                'timestamp': msg.timestamp
                            }

            except pynmea2.ParseError:
                continue
            except Exception:
                continue

    def smooth_coordinates(self, lat, lon):
        self.lat_buffer.append(lat)
        self.lon_buffer.append(lon)
        return statistics.mean(self.lat_buffer), statistics.mean(self.lon_buffer)

    def get_location(self):
        with self.lock:
            return self.latest_location

    def close(self):
        self.running = False
        self.ser.close()
