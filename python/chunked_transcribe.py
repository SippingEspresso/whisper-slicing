#!/usr/bin/env python3
"""
Parallel Whisper Transcription Client with Overlap-Aware Slicing and Stitching.
Dispatches chunked audio requests concurrently across a multi-worker cluster
to achieve multi-GPU speedups without clipping words or sentences at boundaries.
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
import time
import requests


def probe_duration(file_path: str) -> float:
    """Extract audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed to read {file_path}: {res.stderr}")
    return float(res.stdout.strip())


def slice_audio_chunk(input_file: str, start: float, duration: float, output_file: str):
    """
    Slice audio segment using ffmpeg with high-speed stream copying.
    Falls back to re-encoding if container stream copy is not supported.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", input_file,
        "-c", "copy",
        output_file
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Fallback to audio re-encode (fast mp3/wav if stream copy failed on keyframes)
        cmd_fallback = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", input_file,
            "-ar", "16000", "-ac", "1",
            output_file
        ]
        res_fb = subprocess.run(cmd_fallback, capture_output=True, text=True)
        if res_fb.returncode != 0:
            raise RuntimeError(f"ffmpeg slicing failed: {res_fb.stderr}")


def transcribe_worker(worker_id: int, chunk_path: str, server_url: str, model_name: str, language: str = None) -> dict:
    """Send a single chunk to the whisper load balancer / server."""
    endpoint = f"{server_url.rstrip('/')}/v1/audio/transcriptions"
    
    data = {
        "model": model_name,
        "response_format": "verbose_json"
    }
    if language:
        data["language"] = language

    with open(chunk_path, "rb") as f:
        files = {"file": (os.path.basename(chunk_path), f, "audio/octet-stream")}
        t0 = time.time()
        resp = requests.post(endpoint, data=data, files=files, timeout=900)
        elapsed = time.time() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"Worker {worker_id} failed with HTTP {resp.status_code}: {resp.text}")

    return {
        "worker_id": worker_id,
        "elapsed": elapsed,
        "data": resp.json()
    }


def format_srt_time(seconds: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments: list) -> str:
    """Generate SRT formatted subtitles from segments."""
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start_str = format_srt_time(seg["start"])
        end_str = format_srt_time(seg["end"])
        text = seg["text"].strip()
        srt_lines.append(f"{i}\n{start_str} --> {end_str}\n{text}\n")
    return "\n".join(srt_lines)


def run_parallel_transcription(
    input_file: str,
    server_url: str = "http://127.0.0.1:8090",
    num_workers: int = 4,
    overlap_sec: float = 2.0,
    model_name: str = "Systran/faster-whisper-large-v3",
    language: str = None,
    output_prefix: str = None
):
    print(f"\n=======================================================")
    print(f"🚀 Parallel Whisper Transcription")
    print(f"=======================================================")
    print(f"📁 Input File      : {input_file}")
    print(f"🌐 Server Cluster   : {server_url}")
    print(f"⚡ Workers         : {num_workers}")
    print(f"⏱️  Overlap Window  : {overlap_sec}s")
    print(f"🤖 Model           : {model_name}")
    print(f"=======================================================\n")

    t_start = time.time()

    # Step 1: Probe audio
    total_duration = probe_duration(input_file)
    print(f"📊 Audio Duration  : {total_duration:.1f}s ({total_duration/60:.2f} minutes)")

    # Step 2: Calculate chunk boundaries
    chunk_nominal_len = total_duration / num_workers
    chunk_specs = []

    for i in range(num_workers):
        nominal_start = i * chunk_nominal_len
        nominal_end = (i + 1) * chunk_nominal_len if i < num_workers - 1 else total_duration

        # Slicing with overlap buffer
        slice_start = max(0.0, nominal_start - (overlap_sec if i > 0 else 0.0))
        slice_end = min(total_duration, nominal_end + (overlap_sec if i < num_workers - 1 else 0.0))
        slice_duration = slice_end - slice_start

        chunk_specs.append({
            "id": i,
            "nominal_start": nominal_start,
            "nominal_end": nominal_end,
            "slice_start": slice_start,
            "slice_end": slice_end,
            "slice_duration": slice_duration
        })

    # Step 3: Fast slice chunks to temporary directory
    temp_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
    ext = os.path.splitext(input_file)[1] or ".m4a"
    chunk_paths = []

    t_slice_start = time.time()
    for spec in chunk_specs:
        out_path = os.path.join(temp_dir, f"chunk_{spec['id']:02d}{ext}")
        slice_audio_chunk(input_file, spec["slice_start"], spec["slice_duration"], out_path)
        spec["file_path"] = out_path
        chunk_paths.append(out_path)
        print(f"✂️  Chunk {spec['id']}: [{spec['slice_start']:.1f}s -> {spec['slice_end']:.1f}s] "
              f"(len: {spec['slice_duration']:.1f}s, nominal: {spec['nominal_start']:.1f}s - {spec['nominal_end']:.1f}s)")

    t_slice_total = time.time() - t_slice_start
    print(f"✨ Slicing completed in {t_slice_total:.2f}s\n")

    # Step 4: Dispatch in parallel across workers
    print(f"🔥 Dispatching {num_workers} concurrent requests to cluster...")
    t_transcribe_start = time.time()
    results = [None] * num_workers

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(
                transcribe_worker,
                spec["id"],
                spec["file_path"],
                server_url,
                model_name,
                language
            ): spec for spec in chunk_specs
        }

        for future in concurrent.futures.as_completed(future_map):
            spec = future_map[future]
            worker_id = spec["id"]
            try:
                res = future.result()
                results[worker_id] = res
                print(f"  ✅ Worker {worker_id} finished in {res['elapsed']:.1f}s "
                      f"({spec['slice_duration']/res['elapsed']:.1f}× realtime)")
            except Exception as e:
                print(f"  ❌ Worker {worker_id} failed: {e}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise

    t_transcribe_total = time.time() - t_transcribe_start
    print(f"\n⚡ All parallel transcriptions finished in {t_transcribe_total:.2f}s")

    # Step 5: Clean up temp chunks
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Step 6: Overlap Deduplication & Stitching
    print("🧩 Merging and stitching segment boundaries...")
    merged_segments = []
    seg_counter = 1

    for spec, res in zip(chunk_specs, results):
        worker_data = res["data"]
        raw_segments = worker_data.get("segments", [])
        slice_start = spec["slice_start"]
        nominal_start = spec["nominal_start"]
        nominal_end = spec["nominal_end"]
        is_first = (spec["id"] == 0)
        is_last = (spec["id"] == num_workers - 1)

        for seg in raw_segments:
            abs_start = slice_start + seg["start"]
            abs_end = slice_start + seg["end"]
            midpoint = (abs_start + abs_end) / 2.0

            # Overlap boundary filtering:
            # Drop segments belonging to the previous chunk's nominal window
            if not is_first and midpoint < nominal_start:
                continue
            # Drop segments belonging to the next chunk's nominal window
            if not is_last and midpoint >= nominal_end:
                continue

            merged_segments.append({
                "id": seg_counter,
                "start": round(abs_start, 3),
                "end": round(abs_end, 3),
                "text": seg["text"]
            })
            seg_counter += 1

    # Reconstruct full text
    full_text = "".join(seg["text"] for seg in merged_segments).strip()
    total_wall_clock = time.time() - t_start
    realtime_factor = total_duration / total_wall_clock if total_wall_clock > 0 else 0

    print(f"\n=======================================================")
    print(f"🎉 Transcription Complete!")
    print(f"=======================================================")
    print(f"⏱️  Wall-Clock Time   : {total_wall_clock:.2f}s ({total_wall_clock/60:.2f} min)")
    print(f"🚀 Effective Speed    : {realtime_factor:.1f}× realtime")
    print(f"📝 Total Segments     : {len(merged_segments)}")
    print(f"🔤 Total Characters   : {len(full_text)}")
    print(f"=======================================================\n")

    # Step 7: Write Output Files
    if not output_prefix:
        base, _ = os.path.splitext(input_file)
        output_prefix = base

    txt_file = f"{output_prefix}.txt"
    srt_file = f"{output_prefix}.srt"

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(full_text + "\n")
    print(f"📄 Transcript saved to: {txt_file}")

    with open(srt_file, "w", encoding="utf-8") as f:
        f.write(generate_srt(merged_segments))
    print(f"🎬 Subtitles saved to : {srt_file}")

    return {
        "text": full_text,
        "segments": merged_segments,
        "duration": total_duration,
        "wall_clock": total_wall_clock,
        "speedup": realtime_factor
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel Whisper Transcription with Overlap Stitching")
    parser.add_argument("input_file", help="Path to input audio file (m4a, mp3, wav, flac, etc.)")
    parser.add_argument("--url", default="http://127.0.0.1:8090", help="Whisper server/load-balancer URL (default: http://127.0.0.1:8090)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel chunks/workers (default: 4)")
    parser.add_argument("--overlap", type=float, default=2.0, help="Overlap window in seconds at chunk boundaries (default: 2.0)")
    parser.add_argument("--model", default="Systran/faster-whisper-large-v3", help="Model name (default: Systran/faster-whisper-large-v3)")
    parser.add_argument("--language", "-l", default=None, help="Language code (e.g. en, es, fr, de)")
    parser.add_argument("--output", "-o", default=None, help="Output file prefix (without extension)")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    run_parallel_transcription(
        input_file=args.input_file,
        server_url=args.url,
        num_workers=args.workers,
        overlap_sec=args.overlap,
        model_name=args.model,
        language=args.language,
        output_prefix=args.output
    )


if __name__ == "__main__":
    main()
