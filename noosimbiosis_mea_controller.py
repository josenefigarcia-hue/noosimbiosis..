#!/usr/bin/env python3
# noosimbiosis_mea_controller.py
# Script operativo para Fase 2 (simulación). Reemplazar stubs MEA_* por API real.
import numpy as np
import time
import json
import hashlib
import uuid
import math
import logging
from multiprocessing import Process, Queue, Event
from typing import Dict, Any, List, Tuple
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------
# HARD CONSTRAINTS / PARAMS
# ---------------------------
FREQ_CONVULSION_HZ = 300       # banda alta para convulsión (heurístico)
BURST_RATE_THRESHOLD = 3.0     # veces el baseline
LATENCY_KILL_MAX_MS = 40       # objetivo de latencia del sistema (simulado)
MAX_AMPLITUD_SAFE = 0.8        # ejemplo de límite absoluto (V o µA según calibración)
MAX_ALPHA_INVITRO = 0.2
WINDOW_SIZE_MS = 250           # ventana de análisis
ACQUISITION_RATE_HZ = 40000    # muestreo simulado (40 kHz)
DURATION_MAX_SECS = 60         # duración máxima de sesión (segundos)

# Baselines (deberían calibrarse en laboratorio)
BASELINE_BURST_RATE = 0.5
BASELINE_DC_LEVEL = 0.0

# ---------------------------
# STUB: MEA API SIMULATOR
# Replace these with real device API calls
# ---------------------------
class MEAStimulatorStub:
    def __init__(self):
        self.running = False
    def load_sequence(self, seq):
        logging.info("MEASTIM: Secuencia cargada. Pulsos: %d", len(seq))
        self.seq = seq
    def start(self):
        self.running = True
        logging.info("MEASTIM: Inicio de estimulación (simulado).")
    def stop(self):
        if self.running:
            self.running = False
            logging.info("MEASTIM: Parada de estimulación (simulado).")
    def kill_switch_hardware_trigger(self):
        # In real hardware, trigger a physical relay/kill
        self.running = False
        logging.warning("MEASTIM: Kill-switch físico TRIGGERED (simulado).")
    def send_pulse(self, pulse):
        # Placeholder: in real system send single pulse
        logging.debug("MEASTIM: Enviando pulso %s", pulse)

class MEAAcquisitionStub:
    def __init__(self):
        self.running = False
    def start_real_time_acquisition(self):
        self.running = True
        logging.info("MEAACQ: Inicio de adquisición (simulado).")
    def stop(self):
        self.running = False
        logging.info("MEAACQ: Stop adquisición (simulado).")
    def get_latest_window(self, duration_ms=WINDOW_SIZE_MS) -> np.ndarray:
        # Return simulated window of activity (noise + occasional bursts)
        n_samples = int(ACQUISITION_RATE_HZ * (duration_ms / 1000.0))
        # Simulate baseline noise
        data = 0.05 * np.random.randn(n_samples)
        # Occasionally simulate bursts/high mean
        if np.random.rand() > 0.995:
            data += 1.0 * np.random.rand(n_samples)  # synthetic burst
        return data

# Instantiate stubs (replace when integrating with real API)
MEA_STIMULATOR = MEAStimulatorStub()
MEA_ACQUISITION = MEAAcquisitionStub()

# ---------------------------
# Utility: metrics & detectors
# ---------------------------
def calculate_power_in_band(data: np.ndarray, center_hz: float, bandwidth_hz: float = 100.0) -> float:
    """
    Heuristic: estimate power around center_hz using FFT. Works on simulated data.
    """
    if data.size == 0:
        return 0.0
    # compute FFT power spectral density (simple)
    freqs = np.fft.rfftfreq(data.size, d=1.0/ACQUISITION_RATE_HZ)
    psd = np.abs(np.fft.rfft(data))**2
    # integrate band
    band_mask = (freqs >= max(1, center_hz - bandwidth_hz/2)) & (freqs <= center_hz + bandwidth_hz/2)
    if not np.any(band_mask):
        return 0.0
    return float(psd[band_mask].mean())

def calculate_burst_rate(data: np.ndarray, threshold: float = 0.2) -> float:
    """
    Very simple burst rate heuristic: count windows where mean > threshold.
    """
    if data.size == 0:
        return 0.0
    window_len = int(ACQUISITION_RATE_HZ * 0.05)  # 50 ms subwindows
    if window_len == 0:
        return 0.0
    n_windows = data.size // window_len
    if n_windows == 0:
        return 0.0
    counts = 0
    for i in range(n_windows):
        w = data[i*window_len:(i+1)*window_len]
        if np.abs(w.mean()) > threshold:
            counts += 1
    return counts / (data.size / ACQUISITION_RATE_HZ)  # bursts per second approx.

# ---------------------------
# Acquisition/Analysis Thread (Hilo A)
# ---------------------------
def acquisition_and_analysis_thread(q_data: Queue, e_kill: Event, q_telemetry: Queue):
    logging.info("Hilo A: Monitor de seguridad iniciado.")
    last_check = time.time()
    # To measure kill latency in a simulated way
    while not e_kill.is_set():
        # Acquire a window from hardware (or stub)
        window = MEA_ACQUISITION.get_latest_window(WINDOW_SIZE_MS)
        # Push raw window for potential post-hoc logging
        try:
            q_data.put_nowait(window)
        except Exception:
            pass

        # 1. Power in high band (convulsión)
        power_h = calculate_power_in_band(window, FREQ_CONVULSION_HZ, bandwidth_hz=200.0)
        # heuristic threshold (tune in lab)
        if power_h > 1e3:  # placeholder threshold for strong HF power
            logging.warning("Hilo A: Alta potencia en banda alta detectada (power=%s).", power_h)
            e_kill.set()
            q_telemetry.put({"type":"INCIDENT","code":"HF_POWER","power":power_h,"t":time.time()})
            break

        # 2. Burst rate detection
        burst_rate = calculate_burst_rate(window)
        if burst_rate > BASELINE_BURST_RATE * BURST_RATE_THRESHOLD:
            logging.warning("Hilo A: Burst rate excesivo detectado (rate=%s).", burst_rate)
            e_kill.set()
            q_telemetry.put({"type":"INCIDENT","code":"BURST_RATE","burst_rate":burst_rate,"t":time.time()})
            break

        # 3. DC shift detection
        dc_level = float(window.mean())
        if abs(dc_level - BASELINE_DC_LEVEL) > 0.5:  # 0.5 mV heuristic
            logging.warning("Hilo A: DC shift detectado (dc=%s).", dc_level)
            e_kill.set()
            q_telemetry.put({"type":"INCIDENT","code":"DC_SHIFT","dc":dc_level,"t":time.time()})
            break

        # sleep a small fraction to allow responsiveness
        time.sleep(WINDOW_SIZE_MS / 1000.0 * 0.5)

    logging.info("Hilo A: Terminando monitor (kill=%s).", e_kill.is_set())

# ---------------------------
# Control / Stimulation Thread (Hilo B)
# ---------------------------
def control_and_stimulation_thread(q_data: Queue, e_kill: Event, sequence: List[Dict], session_meta: Dict, q_telemetry: Queue):
    start_time = time.time()
    logging.info("Hilo B: Inicio de transducción (α=%s).", session_meta.get("alpha_target"))
    MEA_STIMULATOR.load_sequence(sequence)
    MEA_ACQUISITION.start_real_time_acquisition()
    MEA_STIMULATOR.start()
    try:
        for pulse in sequence:
            if e_kill.is_set():
                logging.warning("Hilo B: Kill-switch detectado, abortando secuencia.")
                break
            # In real device replace send_pulse with API call (non-blocking if possible)
            MEA_STIMULATOR.send_pulse(pulse)
            # Simulate time for pulse duration and provide data to queue for monitoring
            duration_s = pulse.get('duration_ms', 10) / 1000.0
            # Simulate acquisition chunk during pulse
            n_samples = int(ACQUISITION_RATE_HZ * duration_s)
            simulated_window = 0.05 * np.random.randn(n_samples)
            try:
                q_data.put_nowait(simulated_window)
            except Exception:
                pass
            time.sleep(duration_s)
        logging.info("Hilo B: Secuencia terminada (kill=%s).", e_kill.is_set())
    finally:
        # ensure hardware stop
        MEA_STIMULATOR.stop()
        MEA_ACQUISITION.stop()
        session_meta['kill_switch_triggered'] = e_kill.is_set()
        session_meta['incident_flag'] = e_kill.is_set()
        session_meta['end_timestamp_utc'] = time.time()
        # Save telemetry incident if present
        telemetry_items = []
        while not q_telemetry.empty():
            telemetry_items.append(q_telemetry.get())
        session_meta['telemetry'] = telemetry_items
        # checksum & save
        checksum = hashlib.sha256(json.dumps(session_meta, sort_keys=True).encode()).hexdigest()
        session_meta['checksum'] = checksum
        logfile = f"log_{session_meta['session_id']}.json"
        with open(logfile, 'w') as f:
            json.dump(session_meta, f, indent=2)
        logging.info("Hilo B: Telemetría guardada en %s (checksum=%s).", logfile, checksum)

# ---------------------------
# Precompile sequence and launch
# ---------------------------
def precompile_and_launch(alpha_target: float, Q_script_data: np.ndarray) -> Dict[str, Any]:
    """
    Q_script_data: array of tuples (electrode_id, duration_ms, amplitude_target)
    amplitude_target should be in device-native units; scaling applied below.
    """
    session_id = str(uuid.uuid4())
    checksum_q = hashlib.sha256(Q_script_data.tobytes()).hexdigest()

    # Pre-compile and validate
    compiled_sequence = []
    for i, row in enumerate(Q_script_data):
        electrode, duration_ms, amplitude_target = int(row[0]), float(row[1]), float(row[2])
        amplitude_final = min(amplitude_target * (alpha_target / MAX_ALPHA_INVITRO), MAX_AMPLITUD_SAFE)
        # Hard assert
        if amplitude_final > MAX_AMPLITUD_SAFE + 1e-9:
            raise AssertionError("VALIDATION ERROR: Amplitude exceeds safe limit.")
        compiled_sequence.append({
            'id': i,
            'electrode': electrode,
            'amplitude_v': amplitude_final,
            'duration_ms': duration_ms
        })

    # concurrency structures
    q_data = Queue(maxsize=10)
    q_telemetry = Queue()
    e_kill = Event()
    meta_data: Dict[str, Any] = {
        "session_id": session_id,
        "alpha_target": alpha_target,
        "checksum_Q": checksum_q,
        "timestamp_utc": time.time(),
        "incident_flag": False,
        "kill_switch_triggered": False,
        "mean_burst_rate": 0.0
    }

    # start processes
    p_analysis = Process(target=acquisition_and_analysis_thread, args=(q_data, e_kill, q_telemetry))
    p_control = Process(target=control_and_stimulation_thread, args=(q_data, e_kill, compiled_sequence, meta_data, q_telemetry))

    p_analysis.start()
    # small delay to ensure monitor is active before stimulation
    time.sleep(0.1)
    p_control.start()

    # enforce max duration
    max_wait = DURATION_MAX_SECS + 5
    p_control.join(timeout=max_wait)
    if p_control.is_alive():
        logging.warning("Main: Control thread excedió tiempo máximo; forzando terminación.")
        e_kill.set()
        p_control.terminate()
    # analysis process may still be alive; terminate gracefully
    if p_analysis.is_alive():
        p_analysis.terminate()

    # read back meta JSON file
    logfile = f"log_{meta_data['session_id']}.json"
    if os.path.exists(logfile):
        with open(logfile, 'r') as f:
            saved_meta = json.load(f)
        return saved_meta
    return meta_data

# ---------------------------
# Example usage (simulation)
# ---------------------------
if __name__ == "__main__":
    # Simulated Q_script: 5 pulses across electrodes 0..4
    Q = np.array([
        (0, 10.0, 0.5),
        (1, 10.0, 0.4),
        (2, 20.0, 0.6),
        (3, 15.0, 0.3),
        (4, 12.0, 0.45)
    ], dtype=float)

    alpha = 0.1  # safe testing alpha
    logging.info("MAIN: Lanzando sesión simulada.")
    meta = precompile_and_launch(alpha, Q)
    logging.info("MAIN: Sesión finalizada. Meta: %s", json.dumps(meta, indent=2))
