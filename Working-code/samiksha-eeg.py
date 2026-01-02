"""
EEG Recorder - Saves to edf file
Visualizes real-time EEG with signal quality indicator
LED blinks slow = good signal, fast = bad signal
"""

import serial
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque
import matplotlib.animation as animation
import numpy as np
from datetime import datetime
import struct

# Configuration
PORT = 'COM5'
BAUD = 115200
SAMPLE_RATE = 256
EDF_FILE = r"C:\Users\bcsam\OneDrive\Desktop\hardware\output.edf"

# Data storage
eeg_buffer = deque(maxlen=SAMPLE_RATE * 10)  # 10 sec display
all_eeg_data = []
quality_buffer = deque(maxlen=SAMPLE_RATE * 10)
data_started = False
sample_count = 0
start_time = None

# Serial
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    ser.flushInput()
    print(f"Connected to {PORT}")
except Exception as e:
    print(f"Could not open {PORT}: {e}")
    exit(1)


class EDFWriter:
    def __init__(self, filename, patient_name="Samiksha", sample_rate=256):
        self.filename = filename
        self.patient_name = patient_name
        self.sample_rate = sample_rate
        self.data = []

    def add_samples(self, samples):
        self.data.extend(samples)

    def write(self):
        if not self.data:
            print("No data to write!")
            return False

        n_records = len(self.data) // self.sample_rate
        if n_records == 0:
            print("Not enough data (need at least 1 second)")
            return False

        n_samples = n_records * self.sample_rate
        data = self.data[:n_samples]

        data_array = np.array(data, dtype=np.float64)
        physical_min = -500.0
        physical_max = 500.0
        digital_min = -32768
        digital_max = 32767

        data_array = np.clip(data_array, physical_min, physical_max)
        scale = (digital_max - digital_min) / (physical_max - physical_min)
        offset = digital_max - scale * physical_max
        digital_data = (data_array * scale + offset).astype(np.int16)

        with open(self.filename, 'wb') as f:
            now = datetime.now()

            # Header
            f.write(b'0       ')
            patient = f"X X X {self.patient_name}".ljust(80)[:80]
            f.write(patient.encode('ascii'))
            recording = f"Startdate {now.strftime('%d-%b-%Y').upper()} X EEG_Recording".ljust(80)[:80]
            f.write(recording.encode('ascii'))
            f.write(now.strftime('%d.%m.%y').encode('ascii'))
            f.write(now.strftime('%H.%M.%S').encode('ascii'))
            f.write(str(256 + 256).ljust(8).encode('ascii'))
            f.write(b'EDF+C'.ljust(44))
            f.write(str(n_records).ljust(8).encode('ascii'))
            f.write(b'1       ')
            f.write(b'1   ')

            # Signal header
            f.write(b'EEG Fp1         ')
            f.write(b'BioAmp EXG Pill AgAgCl electrode'.ljust(80))
            f.write(b'uV      ')
            f.write(str(physical_min).ljust(8)[:8].encode('ascii'))
            f.write(str(physical_max).ljust(8)[:8].encode('ascii'))
            f.write(str(digital_min).ljust(8)[:8].encode('ascii'))
            f.write(str(digital_max).ljust(8)[:8].encode('ascii'))
            f.write(b'HP:0.5Hz LP:45Hz Notch:50,60Hz'.ljust(80))
            f.write(str(self.sample_rate).ljust(8).encode('ascii'))
            f.write(b' ' * 32)

            # Data
            for i in range(n_records):
                start = i * self.sample_rate
                end = start + self.sample_rate
                f.write(digital_data[start:end].tobytes())

        print(f"\nEDF saved: {self.filename}")
        print(f"Duration: {n_records} seconds, {n_samples} samples")
        return True


edf_writer = EDFWriter(EDF_FILE, patient_name="Samiksha", sample_rate=SAMPLE_RATE)

# Plot setup
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 0.5])

# EEG waveform
ax_eeg = fig.add_subplot(gs[0])
ax_eeg.set_title('EEG Signal - Recording to output.edf', fontsize=14, fontweight='bold', color='cyan')
line_eeg, = ax_eeg.plot([], [], 'lime', linewidth=0.8)
ax_eeg.set_xlim(0, 10)
ax_eeg.set_ylim(-100, 100)
ax_eeg.set_ylabel('Amplitude (ÂµV)')
ax_eeg.set_xlabel('Time (seconds)')
ax_eeg.grid(True, alpha=0.3)

# Signal quality bar
ax_quality = fig.add_subplot(gs[1])
ax_quality.set_title('Signal Quality', fontsize=12, fontweight='bold')
quality_line, = ax_quality.plot([], [], 'cyan', linewidth=2)
ax_quality.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='Threshold')
ax_quality.set_xlim(0, 10)
ax_quality.set_ylim(0, 1.1)
ax_quality.set_ylabel('Quality')
ax_quality.legend(loc='upper right')
ax_quality.grid(True, alpha=0.3)

# Status bar
ax_status = fig.add_subplot(gs[2])
ax_status.axis('off')
status_text = ax_status.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center',
                              fontsize=14, fontweight='bold', color='white',
                              transform=ax_status.transAxes)

# LED indicator
led_indicator = ax_status.text(0.9, 0.5, 'â—', ha='center', va='center',
                                fontsize=24, color='gray',
                                transform=ax_status.transAxes)

plt.tight_layout()


def update(frame):
    global data_started, sample_count, start_time

    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()

            if line.startswith('#'):
                print(line)
                continue

            if line == 'DATA_START':
                data_started = True
                start_time = datetime.now()
                print("\n>>> Recording started! <<<")
                continue

            if not data_started:
                continue

            # Parse: timestamp, eeg_uV, quality
            parts = line.split(',')
            if len(parts) >= 3:
                eeg = float(parts[1])
                quality = int(parts[2])

                eeg_buffer.append(eeg)
                quality_buffer.append(quality)
                all_eeg_data.append(eeg)
                sample_count += 1

        except:
            pass

    # Update plots
    if len(eeg_buffer) > 10:
        t = np.linspace(0, len(eeg_buffer) / SAMPLE_RATE, len(eeg_buffer))
        line_eeg.set_data(t, list(eeg_buffer))
        quality_line.set_data(t, list(quality_buffer))

        ax_eeg.set_xlim(0, max(10, t[-1]))
        ax_quality.set_xlim(0, max(10, t[-1]))

        # Auto-scale EEG
        eeg_list = list(eeg_buffer)
        ymax = max(abs(min(eeg_list)), abs(max(eeg_list))) * 1.2
        ymax = max(ymax, 50)
        ax_eeg.set_ylim(-ymax, ymax)

    # Update status
    duration = sample_count / SAMPLE_RATE if sample_count > 0 else 0
    recent_quality = list(quality_buffer)[-100:] if quality_buffer else [0]
    avg_quality = sum(recent_quality) / len(recent_quality)

    status = f"Recording: {duration:.1f}s | Samples: {sample_count} | "
    status += f"Quality: {avg_quality*100:.0f}%"

    if avg_quality > 0.5:
        status += " | GOOD SIGNAL (slow LED)"
        led_indicator.set_color('lime')
        ax_eeg.set_facecolor('#001100')
    else:
        status += " | POOR SIGNAL (fast LED)"
        led_indicator.set_color('red')
        ax_eeg.set_facecolor('#110000')

    status_text.set_text(status)

    return line_eeg, quality_line, status_text, led_indicator


def on_close(event):
    print("\n" + "="*50)
    print("Saving EEG data to output.edf...")
    print("="*50)

    if all_eeg_data:
        edf_writer.add_samples(all_eeg_data)
        edf_writer.write()
    else:
        print("No data recorded!")


fig.canvas.mpl_connect('close_event', on_close)

print("\n" + "="*60)
print("   EEG RECORDER - Saving to output.edf")
print("="*60)
print(f"\n   Output: {EDF_FILE}")
print("   Sample Rate: 256 Hz")
print("\n   LED Indicators:")
print("     SLOW blink = Good EEG signal")
print("     FAST blink = Poor/no signal")
print("\n   Close window to save EDF file")
print("="*60 + "\n")

ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.show()

ser.close()