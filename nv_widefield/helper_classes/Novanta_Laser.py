import time
import threading
import collections
import tkinter as tk
from tkinter import ttk, messagebox
import serial
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LaserInterface:
    def __init__(self, port='COM3', baudrate=9600, timeout=1.0):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.lock = threading.Lock()

    def send_command(self, cmd: str) -> str:
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.ser.write(f"{cmd.strip()}\r".encode('ascii'))
            time.sleep(0.1)
            response = self.ser.read_all().decode('ascii', errors='ignore').strip()
            # print(f"[DEBUG] CMD: {cmd.strip()} -> RAW RESP: '{repr(response)}'")
            return response

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


class LaserDashboardGUI:
    def __init__(self, root, com_port='COM3'):
        self.root = root
        self.root.title("Laser Control & Telemetry Dashboard")
        self.root.geometry("950x600")

        try:
            self.laser = LaserInterface(port=com_port)
        except Exception as e:
            messagebox.showerror("Serial Error", f"Unable to connect to {com_port}: {e}")
            self.root.destroy()
            return

        self.max_points = 120
        self.time_data = collections.deque(maxlen=self.max_points)
        self.power_data = collections.deque(maxlen=self.max_points)
        self.head_temp_data = collections.deque(maxlen=self.max_points)
        self.psu_temp_data = collections.deque(maxlen=self.max_points)
        self.start_time = time.time()

        self._init_ui()

        self.is_running = True
        self.poll_thread = threading.Thread(target=self._poll_telemetry, daemon=True)
        self.poll_thread.start()

    def _init_ui(self):
        ctrl_frame = ttk.LabelFrame(self.root, text="Controls")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        dash_frame = ttk.Frame(self.root)
        dash_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Button(ctrl_frame, text="Laser ON", command=self._turn_on).pack(fill=tk.X, pady=4, padx=5)
        ttk.Button(ctrl_frame, text="Laser OFF", command=self._turn_off).pack(fill=tk.X, pady=4, padx=5)
        ttk.Button(ctrl_frame, text="Set Mode: POWER", command=self._set_power_mode).pack(fill=tk.X, pady=4, padx=5)

        pwr_frame = ttk.Frame(ctrl_frame)
        pwr_frame.pack(fill=tk.X, pady=10, padx=5)
        ttk.Label(pwr_frame, text="Set Power (mW):").pack(anchor=tk.W)
        self.pwr_entry = ttk.Entry(pwr_frame, width=10)
        self.pwr_entry.insert(0, "5")
        self.pwr_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(pwr_frame, text="Apply", command=self._set_power_value).pack(side=tk.LEFT)

        tele_frame = ttk.LabelFrame(ctrl_frame, text="Readouts")
        tele_frame.pack(fill=tk.X, pady=15, padx=5)

        self.lbl_power = ttk.Label(tele_frame, text="Power: -- mW", font=('Consolas', 10))
        self.lbl_power.pack(anchor=tk.W, pady=2)
        self.lbl_head_temp = ttk.Label(tele_frame, text="Head Temp: -- °C", font=('Consolas', 10))
        self.lbl_head_temp.pack(anchor=tk.W, pady=2)
        self.lbl_psu_temp = ttk.Label(tele_frame, text="PSU Temp: -- °C", font=('Consolas', 10))
        self.lbl_psu_temp.pack(anchor=tk.W, pady=2)

        self.fig, (self.ax_pwr, self.ax_temp) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        self.fig.tight_layout(pad=3.0)

        self.line_pwr, = self.ax_pwr.plot([], [], 'r-', label="Power (mW)")
        self.ax_pwr.set_ylabel("Power (mW)")
        self.ax_pwr.legend(loc="upper left")
        self.ax_pwr.grid(True)

        self.line_htemp, = self.ax_temp.plot([], [], 'b-', label="Head Temp (°C)")
        self.line_ptemp, = self.ax_temp.plot([], [], 'g-', label="PSU Temp (°C)")
        self.ax_temp.set_ylabel("Temp (°C)")
        self.ax_temp.set_xlabel("Elapsed Time (s)")
        self.ax_temp.legend(loc="upper left")
        self.ax_temp.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=dash_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _turn_on(self):
        self.laser.send_command("ON")

    def _turn_off(self):
        self.laser.send_command("OFF")

    def _set_power_mode(self):
        self.laser.send_command("CONTROL=POWER")

    def _set_power_value(self):
        val = self.pwr_entry.get().strip()
        if val.isdigit():
            self.laser.send_command(f"POWER={val}")
        else:
            messagebox.showwarning("Invalid Input", "Power setpoint must be an integer (mW).")

    def _parse_float(self, raw_val: str) -> float:
        match = re.search(r"[-+]?\d*\.\d+|\d+", raw_val)
        if match:
            # print("regex matched, raw_val: ", raw_val, "taking float of ", match.group(0))
            return float(match.group(0))
        print("[DEBUG] no regex match, returning 0 for raw_val: ", raw_val)
        return 0.0

    def _poll_telemetry(self):
        while self.is_running:
            pwr_resp = self.laser.send_command("POWER?")
            htemp_resp = self.laser.send_command("LASTEMP?")
            ptemp_resp = self.laser.send_command("PSUTEMP?")

            t_elapsed = time.time() - self.start_time

            pwr = self._parse_float(pwr_resp)
            htemp = self._parse_float(htemp_resp)
            ptemp = self._parse_float(ptemp_resp)

            self.root.after(0, self._update_gui, t_elapsed, pwr, htemp, ptemp)
            time.sleep(1.0)

    def _update_gui(self, t_elapsed, pwr, htemp, ptemp):
        self.lbl_power.config(text=f"Power: {pwr:.1f} mW")
        self.lbl_head_temp.config(text=f"Head Temp: {htemp:.1f} °C")
        self.lbl_psu_temp.config(text=f"PSU Temp: {ptemp:.1f} °C")

        self.time_data.append(t_elapsed)
        self.power_data.append(pwr)
        self.head_temp_data.append(htemp)
        self.psu_temp_data.append(ptemp)

        self.line_pwr.set_data(self.time_data, self.power_data)
        self.line_htemp.set_data(self.time_data, self.head_temp_data)
        self.line_ptemp.set_data(self.time_data, self.psu_temp_data)

        self.ax_pwr.relim()
        self.ax_pwr.autoscale_view()
        self.ax_temp.relim()
        self.ax_temp.autoscale_view()

        self.canvas.draw_idle()

    def shutdown(self):
        self.is_running = False
        self.laser.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LaserDashboardGUI(root, com_port="COM3")
    root.protocol("WM_DELETE_WINDOW", app.shutdown)
    root.mainloop()