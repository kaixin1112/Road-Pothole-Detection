import serial
import pynmea2

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

while True:
    try:
        line = ser.readline().decode('ascii', errors='replace')
        if line.startswith('$GPRMC') or line.startswith('$GPGGA'):
            msg = pynmea2.parse(line)
            if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                print(f"✅ Lat: {msg.latitude:.5f}, Lon: {msg.longitude:.5f}")
    except pynmea2.ParseError:
        continue
    except KeyboardInterrupt:
        break
