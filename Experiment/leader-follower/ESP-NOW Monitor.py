import argparse
import csv
import re
import time
from pathlib import Path

import serial


NUMBER = rb"[-+]?\d+(?:\.\d+)?"
FRAME_PATTERN = re.compile(
    rb"(" + NUMBER + rb")\s*,\s*(" + NUMBER + rb")\s*,\s*(" + NUMBER + rb")\s*,\s*(" + NUMBER + rb")"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Read ID,x,y,direction numeric frames from a noisy serial stream.")
    parser.add_argument("--port", default="COM4", help="Serial port name, default: COM6")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate, default: 115200")
    parser.add_argument("--timeout", type=float, default=0.1, help="Serial read timeout in seconds")
    parser.add_argument("--wake-interval", type=float, default=5.0, help="Seconds between wake-up writes")
    parser.add_argument("--wake-message", default="PING\n", help="Message sent periodically to wake the device")
    parser.add_argument("--csv", default="", help="Optional CSV output path")
    return parser.parse_args()


def open_csv(path):
    if not path:
        return None, None

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["Timestamp(s)", "ID", "X", "Y", "Direction"])
    return csv_file, writer


def main():
    args = parse_args()
    csv_file, csv_writer = open_csv(args.csv)
    wake_bytes = args.wake_message.encode("utf-8", errors="ignore")

    buffer = b""
    last_wake_time = 0.0
    start_time = time.time()

    try:
        with serial.Serial(args.port, args.baudrate, timeout=args.timeout) as ser:
            print(f"[INFO] Opened {args.port} at {args.baudrate} baud.")
            print("[INFO] Waiting for ID,x,y,direction frames. Press Ctrl+C to stop.")

            while True:
                now = time.time()

                if now - last_wake_time >= args.wake_interval:
                    ser.write(wake_bytes)
                    ser.flush()
                    last_wake_time = now

                data = ser.read(256)
                if data:
                    buffer += data

                    while True:
                        match = FRAME_PATTERN.search(buffer)
                        if not match:
                            break

                        robot_id = int(float(match.group(1)))
                        x = float(match.group(2))
                        y = float(match.group(3))
                        direction = float(match.group(4))
                        timestamp = now - start_time
                        print(f"{timestamp:.3f}s  ID={robot_id}, x={x:.6g}, y={y:.6g}, direction={direction:.6g}")

                        if csv_writer:
                            csv_writer.writerow([f"{timestamp:.6f}", robot_id, x, y, direction])
                            csv_file.flush()

                        buffer = buffer[match.end():]

                    if len(buffer) > 1024:
                        buffer = buffer[-1024:]

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
