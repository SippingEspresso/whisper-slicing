# ⚡ whisper-slicing

> High-performance, overlap-aware parallel audio chunking & transcription client in Go for multi-worker Whisper clusters.

[![Go Version](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com/SippingEspresso/whisper-slicing)

`whisper-slicing` splits large audio files (such as 1–2+ hour meetings or podcasts) into parallel chunks and streams them simultaneously to a multi-GPU / multi-worker Whisper cluster (e.g. `faster-whisper-server` behind an Nginx load balancer). 

By adding a configurable **overlap buffer** (default 1.0s) and performing **midpoint timestamp deduplication**, it eliminates boundary clipping and duplicate words, turning a 7–10 minute transcription job into a **~1.5 to 2 minute** turnaround.

---

## 🚀 Key Features

* **Goroutine-Powered Slicing:** Concurrent audio slicing using `ffmpeg` stream copying (<50ms overhead for an 80-minute file).
* **Overlap Buffering (Zero Word Clipping):** Adds configurable pre-roll and post-roll margins (`--overlap 1.0s`) to ensure boundary words and sentences remain intact.
* **Smart Midpoint Deduplication:** Automatically aligns global segment timestamps and deduplicates overlap regions without leaving gaps or repeated phrases.
* **Low Memory & Static Binary:** Instant startup (<5ms) and tiny memory footprint (~6 MB RAM) with zero Python dependencies on client machines.
* **Automatic Output Generation:** Generates both cleaned plain-text transcripts (`.txt`) and subtitle files (`.srt`).

---

## 📊 Benchmark & Performance

Tested on an **80-minute meeting recording** (`2026-08-27 Meeting 1.m4a`):

| Setup | Hardware | Model & Precision | Concurrency | Wall-Clock Time | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Local CPU** | 8 P-core threads | Large-v3-Turbo (int8) | 1 (Serial) | 11m 45s (705s) | ~6.8× realtime |
| **Remote Single GPU** | 1× RTX 3060 (12GB) | Large-v3 (FP16) | 1 (Serial) | 8m 29s (509s) | ~9.5× realtime |
| **Remote 4-Worker (Unchunked)** | 2× RTX 3060 (24GB) | Large-v3 (`int8_float16`) | 1 (Monolithic) | 7m 12s (432s) | ~11.1× realtime |
| **`whisper-slicing` (Go Cluster)** | 2× RTX 3060 (24GB) | Large-v3 (`int8_float16`) | **4 (Parallel)** | **⚡ 1m 50s (110s)** | **🔥 ~44× realtime** |

---

## 🛠️ How It Works

```
                        ┌──> Worker 1 (GPU 0: Port 8001) [00:00 -> 20:01] ──┐
                        │                                                   │
Audio File (80 min)     ├──> Worker 2 (GPU 0: Port 8002) [19:59 -> 40:01] ──┤   Stitcher & Deduplicator
[ Sliced in ~45ms ] ───>│                                                   ├──> (Midpoint Boundary Alignment)
with Overlap Margins    ├──> Worker 3 (GPU 1: Port 8003) [39:59 -> 60:01] ──┤        │
                        │                                                   │        ▼
                        └──> Worker 4 (GPU 1: Port 8004) [59:59 -> 80:00] ──┘   .txt & .srt Files
```

1. **Probe:** Measures exact duration via `ffprobe`.
2. **Parallel Slice:** Computes nominal windows and spawns $N$ goroutines to slice chunks with `+overlap` padding.
3. **Concurrent Dispatch:** Streams multipart form data to the Whisper backend endpoint (`/v1/audio/transcriptions`) concurrently.
4. **Stitch & Deduplicate:** Offsets segment timestamps to global time and resolves overlaps using boundary midpoint assignment.
5. **Export:** Writes `.txt` and `.srt` with aligned timestamps.

---

## 📦 Installation

### Prerequisites
* `ffmpeg` and `ffprobe` installed and in your `PATH`.
* Go 1.22+ (to compile from source).

### Build from Source
```bash
git clone https://github.com/SippingEspresso/whisper-slicing.git
cd whisper-slicing
make build
make install # installs to ~/.local/bin/whisper-slicing
```

---

## 💻 Usage

```bash
# Basic usage (defaults to 4 workers, 1.0s overlap, http://127.0.0.1:8090)
whisper-slicing -input "meeting.m4a"

# Pointing to a remote GPU server over LAN / Tailscale
whisper-slicing \
  -input "meeting.m4a" \
  -url "http://pigeon.local:8090" \
  -workers 4 \
  -overlap 1.0 \
  -model "Systran/faster-whisper-large-v3" \
  -output "transcripts/meeting_summary"
```

### CLI Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-input` | string | `""` | Path to audio file (`.m4a`, `.mp3`, `.wav`, `.flac`, `.aac`, etc.) |
| `-url` | string | `http://127.0.0.1:8090` | Whisper cluster URL or Nginx load balancer endpoint |
| `-workers` | int | `4` | Number of concurrent slices/workers |
| `-overlap` | float | `1.0` | Overlap window in seconds at chunk boundaries |
| `-model` | string | `Systran/faster-whisper-large-v3` | Model identifier expected by Whisper server |
| `-language` | string | `""` | Optional language code (e.g. `en`, `es`, `de`, `fr`) |
| `-output` | string | `""` | Output filename prefix (defaults to audio base name) |

---

## 📄 License

Apache 2.0. See [LICENSE](LICENSE) for details.
