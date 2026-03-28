"""
DSP Chat — Device B: FSK Decoder
=================================
Listens to microphone → detects 1200 Hz (bit 0) and 2200 Hz (bit 1)
→ reconstructs binary → decodes ASCII text

Requirements:
    pip install pyaudio numpy

Run:
    python decoder.py
"""

import struct
import time
import numpy as np

try:
    import pyaudio
except ImportError:
    print("Missing dependency. Please run:  pip install pyaudio numpy")
    raise

# ─── Must match Device A exactly ─────────────────────────────────────────────
FREQ0        = 1200        # Hz — bit 0 (SPACE)
FREQ1        = 2200        # Hz — bit 1 (MARK)
BIT_DUR      = 0.08        # seconds per bit (80 ms)
SAMPLE_RATE  = 44100       # Hz
PREAMBLE_LEN = 8           # sync bits sent before data

# ─── Detection tuning ────────────────────────────────────────────────────────
# How many samples per FFT window (= one bit window)
CHUNK        = int(SAMPLE_RATE * BIT_DUR)   # 3528 samples @ 80ms

# Frequency tolerance: how many Hz either side of target to accept as a match
FREQ_TOL     = 120          # Hz

# Signal threshold: minimum FFT magnitude to count as a real tone (not silence)
MAG_THRESHOLD = 500

# How many consecutive silent chunks before we treat the message as complete
SILENCE_LIMIT = 6

# ─── Colours for terminal output ─────────────────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RED    = '\033[91m'
DIM    = '\033[2m'
RESET  = '\033[0m'
BOLD   = '\033[1m'


def detect_bit(chunk_samples: np.ndarray) -> tuple[int | None, float]:
    """
    Run FFT on a chunk of audio samples.
    Returns (bit, magnitude) where bit is 0, 1, or None (silence / noise).
    """
    # Apply Hann window to reduce spectral leakage
    window     = np.hanning(len(chunk_samples))
    windowed   = chunk_samples * window

    # Real FFT
    fft_vals   = np.fft.rfft(windowed)
    fft_mag    = np.abs(fft_vals)
    freqs      = np.fft.rfftfreq(len(chunk_samples), d=1.0 / SAMPLE_RATE)

    # Find the peak frequency
    peak_idx   = np.argmax(fft_mag)
    peak_freq  = freqs[peak_idx]
    peak_mag   = fft_mag[peak_idx]

    if peak_mag < MAG_THRESHOLD:
        return None, peak_mag   # too quiet — silence or noise

    if abs(peak_freq - FREQ0) <= FREQ_TOL:
        return 0, peak_mag
    elif abs(peak_freq - FREQ1) <= FREQ_TOL:
        return 1, peak_mag
    else:
        return None, peak_mag   # unrecognised frequency


def bits_to_text(bits: list[int]) -> str:
    """
    Convert a flat list of bits → ASCII text.
    Processes 8 bits at a time (MSB first), skips non-printable chars.
    """
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte_bits = bits[i:i + 8]
        value     = int(''.join(str(b) for b in byte_bits), 2)
        if 32 <= value <= 126:          # printable ASCII range
            chars.append(chr(value))
        elif value == 10 or value == 13:
            chars.append('\n')
    return ''.join(chars)


def format_bits(bits: list[int]) -> str:
    """Pretty-print bits grouped by byte."""
    groups = []
    for i in range(0, len(bits), 8):
        groups.append(''.join(str(b) for b in bits[i:i + 8]))
    return ' '.join(groups)


def decode_stream():
    """Main decode loop — reads mic continuously."""
    pa     = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print(f"\n{BOLD}{'─'*52}{RESET}")
    print(f"{BOLD}  DSP Chat — Device B Decoder{RESET}")
    print(f"{'─'*52}")
    print(f"  Bit 0  →  {FREQ0} Hz  (SPACE)")
    print(f"  Bit 1  →  {FREQ1} Hz  (MARK)")
    print(f"  Bit duration  →  {int(BIT_DUR*1000)} ms")
    print(f"  Sample rate   →  {SAMPLE_RATE} Hz")
    print(f"  Chunk size    →  {CHUNK} samples")
    print(f"{'─'*52}")
    print(f"  {GREEN}Listening…{RESET}  (Ctrl+C to stop)\n")

    # State machine
    state          = 'WAITING'   # WAITING → PREAMBLE → RECEIVING → DONE
    preamble_buf   = []          # preamble bit buffer
    data_bits      = []          # decoded data bits
    silence_count  = 0

    try:
        while True:
            raw     = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            bit, mag = detect_bit(samples)

            # ── WAITING: look for preamble start ─────────────────────────
            if state == 'WAITING':
                if bit is not None:
                    print(f"  {YELLOW}⟶ Preamble detected — syncing…{RESET}")
                    state = 'PREAMBLE'
                    preamble_buf = [bit]
                    silence_count = 0

            # ── PREAMBLE: collect sync bits ───────────────────────────────
            elif state == 'PREAMBLE':
                if bit is not None:
                    preamble_buf.append(bit)
                    silence_count = 0
                    if len(preamble_buf) >= PREAMBLE_LEN:
                        print(f"  {YELLOW}⟶ Sync complete — receiving data…{RESET}")
                        print(f"  {DIM}Preamble: {format_bits(preamble_buf[:PREAMBLE_LEN])}{RESET}\n")
                        state     = 'RECEIVING'
                        data_bits = []
                else:
                    silence_count += 1
                    if silence_count > 3:
                        state = 'WAITING'   # lost signal, restart
                        preamble_buf = []

            # ── RECEIVING: decode data bits ───────────────────────────────
            elif state == 'RECEIVING':
                if bit is not None:
                    data_bits.append(bit)
                    silence_count = 0

                    # Live feedback: show each bit as it arrives
                    sym = f"{CYAN}1{RESET}" if bit == 1 else f"{DIM}0{RESET}"
                    print(f"\r  Bits received: {len(data_bits):>4}   "
                          f"last bit: {sym}   "
                          f"({len(data_bits)//8} chars)     ",
                          end='', flush=True)
                else:
                    silence_count += 1
                    if silence_count >= SILENCE_LIMIT:
                        state = 'DONE'

            # ── DONE: decode and print ────────────────────────────────────
            elif state == 'DONE':
                print()   # newline after live bit counter

                if len(data_bits) >= 8:
                    text = bits_to_text(data_bits)

                    print(f"\n  {'─'*46}")
                    print(f"  {GREEN}{BOLD}📨 Message received:{RESET}")
                    print(f"  {BOLD}{text}{RESET}")
                    print(f"  {'─'*46}")
                    print(f"  {DIM}Total bits : {len(data_bits)}{RESET}")
                    print(f"  {DIM}Binary     : {format_bits(data_bits)}{RESET}")
                    print(f"  {'─'*46}\n")
                else:
                    print(f"  {RED}⚠ Too few bits received ({len(data_bits)}) — message lost{RESET}\n")

                # Reset and wait for next transmission
                state         = 'WAITING'
                data_bits     = []
                preamble_buf  = []
                silence_count = 0
                print(f"  {GREEN}Listening…{RESET}\n")

    except KeyboardInterrupt:
        print(f"\n\n  {DIM}Stopped.{RESET}\n")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == '__main__':
    decode_stream()
