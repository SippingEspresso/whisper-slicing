package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

type WhisperSegment struct {
	ID               int     `json:"id"`
	Start            float64 `json:"start"`
	End              float64 `json:"end"`
	Text             string  `json:"text"`
	Tokens           []int   `json:"tokens,omitempty"`
	Temperature      float64 `json:"temperature,omitempty"`
	AvgLogprob       float64 `json:"avg_logprob,omitempty"`
	CompressionRatio float64 `json:"compression_ratio,omitempty"`
	NoSpeechProb     float64 `json:"no_speech_prob,omitempty"`
}

type VerboseJSONResponse struct {
	Task     string           `json:"task"`
	Language string           `json:"language"`
	Duration float64          `json:"duration"`
	Text     string           `json:"text"`
	Segments []WhisperSegment `json:"segments"`
}

type ChunkSpec struct {
	ID            int
	NominalStart  float64
	NominalEnd    float64
	SliceStart    float64
	SliceEnd      float64
	SliceDuration float64
	FilePath      string
	TargetURL     string
}

type WorkerResult struct {
	WorkerID  int
	TargetURL string
	Elapsed   time.Duration
	Data      VerboseJSONResponse
	Err       error
}

func probeDuration(filePath string) (float64, error) {
	cmd := exec.Command("ffprobe", "-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		filePath,
	)
	out, err := cmd.Output()
	if err != nil {
		return 0, fmt.Errorf("ffprobe failed: %w", err)
	}
	s := strings.TrimSpace(string(out))
	dur, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse duration '%s': %w", s, err)
	}
	return dur, nil
}

func sliceAudioChunk(inputFile string, start, duration float64, outputFile string) error {
	// Attempt fast stream-copy slicing
	cmd := exec.Command("ffmpeg", "-y", "-v", "error",
		"-ss", fmt.Sprintf("%.3f", start),
		"-t", fmt.Sprintf("%.3f", duration),
		"-i", inputFile,
		"-c", "copy",
		outputFile,
	)
	if err := cmd.Run(); err == nil {
		return nil
	}

	// Fallback to fast 16kHz mono re-encode if stream copy fails
	fallback := exec.Command("ffmpeg", "-y", "-v", "error",
		"-ss", fmt.Sprintf("%.3f", start),
		"-t", fmt.Sprintf("%.3f", duration),
		"-i", inputFile,
		"-ar", "16000", "-ac", "1",
		outputFile,
	)
	if err := fallback.Run(); err != nil {
		return fmt.Errorf("ffmpeg slicing failed on %s: %w", outputFile, err)
	}
	return nil
}

func transcribeChunk(client *http.Client, spec ChunkSpec, modelName, language string) (VerboseJSONResponse, error) {
	var respData VerboseJSONResponse

	file, err := os.Open(spec.FilePath)
	if err != nil {
		return respData, fmt.Errorf("open chunk file error: %w", err)
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("file", filepath.Base(spec.FilePath))
	if err != nil {
		return respData, fmt.Errorf("create form file error: %w", err)
	}
	if _, err := io.Copy(part, file); err != nil {
		return respData, fmt.Errorf("copy file error: %w", err)
	}

	if err := writer.WriteField("model", modelName); err != nil {
		return respData, err
	}
	if err := writer.WriteField("response_format", "verbose_json"); err != nil {
		return respData, err
	}
	if language != "" {
		if err := writer.WriteField("language", language); err != nil {
			return respData, err
		}
	}

	if err := writer.Close(); err != nil {
		return respData, err
	}

	endpoint := strings.TrimRight(spec.TargetURL, "/") + "/v1/audio/transcriptions"
	req, err := http.NewRequest("POST", endpoint, body)
	if err != nil {
		return respData, fmt.Errorf("new request error: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	res, err := client.Do(req)
	if err != nil {
		return respData, fmt.Errorf("http post error: %w", err)
	}
	defer res.Body.Close()

	respBytes, err := io.ReadAll(res.Body)
	if err != nil {
		return respData, fmt.Errorf("read response body error: %w", err)
	}

	if res.StatusCode != http.StatusOK {
		return respData, fmt.Errorf("server error %d: %s", res.StatusCode, string(respBytes))
	}

	if err := json.Unmarshal(respBytes, &respData); err != nil {
		return respData, fmt.Errorf("json unmarshal error: %w", err)
	}

	return respData, nil
}

func formatSRTTime(seconds float64) string {
	hrs := int(seconds / 3600)
	mins := int((int(seconds) % 3600) / 60)
	secs := int(int(seconds) % 60)
	millis := int((seconds - float64(int(seconds))) * 1000)
	return fmt.Sprintf("%02d:%02d:%02d,%03d", hrs, mins, secs, millis)
}

func generateSRT(segments []WhisperSegment) string {
	var sb strings.Builder
	for i, seg := range segments {
		startStr := formatSRTTime(seg.Start)
		endStr := formatSRTTime(seg.End)
		text := strings.TrimSpace(seg.Text)
		sb.WriteString(fmt.Sprintf("%d\n%s --> %s\n%s\n\n", i+1, startStr, endStr, text))
	}
	return sb.String()
}

func main() {
	inputFile := flag.String("input", "", "Path to audio file (or pass as first argument)")
	serverURL := flag.String("url", "http://127.0.0.1:8090", "Whisper server/load balancer URL (or comma-separated list of worker URLs)")
	numWorkers := flag.Int("workers", 4, "Number of concurrent parallel workers")
	overlapSec := flag.Float64("overlap", 1.0, "Overlap window in seconds at chunk boundaries")
	modelName := flag.String("model", "Systran/faster-whisper-large-v3", "Model name")
	language := flag.String("language", "", "Language code (optional, e.g. 'en')")
	outputPrefix := flag.String("output", "", "Output prefix path (optional)")

	flag.Parse()

	targetFile := *inputFile
	if targetFile == "" && flag.NArg() > 0 {
		targetFile = flag.Arg(0)
	}

	if targetFile == "" {
		fmt.Println("Usage: whisper-slicing [flags] <audio_file>")
		flag.PrintDefaults()
		os.Exit(1)
	}

	if _, err := os.Stat(targetFile); os.IsNotExist(err) {
		fmt.Printf("Error: File '%s' not found.\n", targetFile)
		os.Exit(1)
	}

	// Parse server URLs (supports comma-separated list or single load-balancer URL)
	rawURLs := strings.Split(*serverURL, ",")
	var urls []string
	for _, u := range rawURLs {
		trimmed := strings.TrimSpace(u)
		if trimmed != "" {
			if !strings.HasPrefix(trimmed, "http://") && !strings.HasPrefix(trimmed, "https://") {
				trimmed = "http://" + trimmed
			}
			urls = append(urls, trimmed)
		}
	}
	if len(urls) == 0 {
		urls = []string{"http://127.0.0.1:8090"}
	}

	fmt.Println("=======================================================")
	fmt.Println("⚡ Go Parallel Whisper Transcription Client")
	fmt.Println("=======================================================")
	fmt.Printf("📁 Audio File     : %s\n", targetFile)
	fmt.Printf("🌐 Endpoints      : %s\n", strings.Join(urls, ", "))
	fmt.Printf("🚀 Workers        : %d (Goroutines)\n", *numWorkers)
	fmt.Printf("⏱️  Overlap Window : %.1fs\n", *overlapSec)
	fmt.Printf("🤖 Model          : %s\n", *modelName)
	fmt.Println("=======================================================")

	tStart := time.Now()

	// 1. Probe total audio duration
	totalDuration, err := probeDuration(targetFile)
	if err != nil {
		fmt.Printf("❌ Failed to probe audio: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("📊 Audio Duration : %.1fs (%.2f minutes)\n\n", totalDuration, totalDuration/60.0)

	// 2. Prepare chunk specifications
	nominalChunkLen := totalDuration / float64(*numWorkers)
	chunkSpecs := make([]ChunkSpec, *numWorkers)

	for i := 0; i < *numWorkers; i++ {
		nomStart := float64(i) * nominalChunkLen
		nomEnd := float64(i+1) * nominalChunkLen
		if i == *numWorkers-1 {
			nomEnd = totalDuration
		}

		slStart := nomStart
		if i > 0 {
			slStart = nomStart - *overlapSec
			if slStart < 0 {
				slStart = 0
			}
		}

		slEnd := nomEnd
		if i < *numWorkers-1 {
			slEnd = nomEnd + *overlapSec
			if slEnd > totalDuration {
				slEnd = totalDuration
			}
		}

		targetEndpoint := urls[i%len(urls)]

		chunkSpecs[i] = ChunkSpec{
			ID:            i,
			NominalStart:  nomStart,
			NominalEnd:    nomEnd,
			SliceStart:    slStart,
			SliceEnd:      slEnd,
			SliceDuration: slEnd - slStart,
			TargetURL:     targetEndpoint,
		}
	}

	// 3. Slice chunks in parallel goroutines
	tempDir, err := os.MkdirTemp("", "whisper_go_chunks_*")
	if err != nil {
		fmt.Printf("❌ Failed to create temp directory: %v\n", err)
		os.Exit(1)
	}
	defer os.RemoveAll(tempDir)

	ext := filepath.Ext(targetFile)
	if ext == "" {
		ext = ".m4a"
	}

	fmt.Println("✂️  Slicing chunks in parallel goroutines...")
	tSliceStart := time.Now()

	var sliceWg sync.WaitGroup
	sliceErrors := make([]error, *numWorkers)

	for i := 0; i < *numWorkers; i++ {
		sliceWg.Add(1)
		go func(idx int) {
			defer sliceWg.Done()
			spec := &chunkSpecs[idx]
			outPath := filepath.Join(tempDir, fmt.Sprintf("chunk_%02d%s", spec.ID, ext))
			spec.FilePath = outPath
			if err := sliceAudioChunk(targetFile, spec.SliceStart, spec.SliceDuration, outPath); err != nil {
				sliceErrors[idx] = err
			}
		}(i)
	}
	sliceWg.Wait()

	for i, sErr := range sliceErrors {
		if sErr != nil {
			fmt.Printf("❌ Failed slicing chunk %d: %v\n", i, sErr)
			os.Exit(1)
		}
		spec := chunkSpecs[i]
		fmt.Printf("  Chunk %d: [%.1fs -> %.1fs] (len: %.1fs, nominal: %.1fs - %.1fs) -> %s\n",
			spec.ID, spec.SliceStart, spec.SliceEnd, spec.SliceDuration, spec.NominalStart, spec.NominalEnd, spec.TargetURL)
	}
	fmt.Printf("✨ All chunks sliced in %v\n\n", time.Since(tSliceStart).Round(time.Millisecond))

	// 4. Concurrent HTTP dispatch
	fmt.Printf("🔥 Dispatching %d concurrent HTTP streams...\n", *numWorkers)
	tTranscribeStart := time.Now()

	results := make([]WorkerResult, *numWorkers)
	var transcribeWg sync.WaitGroup

	for i := 0; i < *numWorkers; i++ {
		transcribeWg.Add(1)
		go func(idx int) {
			defer transcribeWg.Done()
			spec := chunkSpecs[idx]

			// Create dedicated client per worker for clean non-blocking connection isolation
			workerClient := &http.Client{
				Timeout: 900 * time.Second,
				Transport: &http.Transport{
					MaxIdleConns:        10,
					MaxIdleConnsPerHost: 5,
					IdleConnTimeout:     90 * time.Second,
					DisableKeepAlives:   false,
				},
			}

			wStart := time.Now()
			data, err := transcribeChunk(workerClient, spec, *modelName, *language)
			wElapsed := time.Since(wStart)

			results[idx] = WorkerResult{
				WorkerID:  spec.ID,
				TargetURL: spec.TargetURL,
				Elapsed:   wElapsed,
				Data:      data,
				Err:       err,
			}

			if err == nil {
				speedup := spec.SliceDuration / wElapsed.Seconds()
				fmt.Printf("  ✅ Worker %d [%s] finished in %v (%.1f× realtime)\n", spec.ID, spec.TargetURL, wElapsed.Round(time.Millisecond), speedup)
			} else {
				fmt.Printf("  ❌ Worker %d [%s] failed: %v\n", spec.ID, spec.TargetURL, err)
			}
		}(i)
	}
	transcribeWg.Wait()

	for _, res := range results {
		if res.Err != nil {
			fmt.Printf("❌ Transcription failed on worker %d: %v\n", res.WorkerID, res.Err)
			os.Exit(1)
		}
	}
	fmt.Printf("⚡ Parallel transcription complete in %v\n\n", time.Since(tTranscribeStart).Round(time.Millisecond))

	// 5. Overlap Deduplication & Stitching
	fmt.Println("🧩 Stitching segment timestamps and deduplicating boundary overlap...")
	var mergedSegments []WhisperSegment
	segCounter := 1

	for i := 0; i < *numWorkers; i++ {
		spec := chunkSpecs[i]
		res := results[i]
		isFirst := (i == 0)
		isLast := (i == *numWorkers-1)

		for _, rawSeg := range res.Data.Segments {
			absStart := spec.SliceStart + rawSeg.Start
			absEnd := spec.SliceStart + rawSeg.End
			midpoint := (absStart + absEnd) / 2.0

			// Boundary filtering against nominal window
			if !isFirst && midpoint < spec.NominalStart {
				continue
			}
			if !isLast && midpoint >= spec.NominalEnd {
				continue
			}

			mergedSegments = append(mergedSegments, WhisperSegment{
				ID:    segCounter,
				Start: absStart,
				End:   absEnd,
				Text:  rawSeg.Text,
			})
			segCounter++
		}
	}

	var fullTextBuilder strings.Builder
	for _, seg := range mergedSegments {
		fullTextBuilder.WriteString(seg.Text)
	}
	fullText := strings.TrimSpace(fullTextBuilder.String())

	totalWallClock := time.Since(tStart)
	realtimeFactor := totalDuration / totalWallClock.Seconds()

	fmt.Println("\n=======================================================")
	fmt.Println("🎉 Transcription Finished Successfully!")
	fmt.Println("=======================================================")
	fmt.Printf("⏱️  Total Wall-Clock Time : %v (%.2f minutes)\n", totalWallClock.Round(time.Millisecond), totalWallClock.Minutes())
	fmt.Printf("🚀 Effective Cluster Speed: %.1f× realtime\n", realtimeFactor)
	fmt.Printf("📝 Total Merged Segments  : %d\n", len(mergedSegments))
	fmt.Printf("🔤 Total Character Count  : %d\n", len(fullText))
	fmt.Println("=======================================================")

	// 6. Write Outputs
	prefix := *outputPrefix
	if prefix == "" {
		prefix = strings.TrimSuffix(targetFile, filepath.Ext(targetFile))
	}

	txtFile := prefix + ".txt"
	srtFile := prefix + ".srt"

	if err := os.WriteFile(txtFile, []byte(fullText+"\n"), 0644); err != nil {
		fmt.Printf("❌ Failed to write %s: %v\n", txtFile, err)
	} else {
		fmt.Printf("📄 Text transcript : %s\n", txtFile)
	}

	if err := os.WriteFile(srtFile, []byte(generateSRT(mergedSegments)), 0644); err != nil {
		fmt.Printf("❌ Failed to write %s: %v\n", srtFile, err)
	} else {
		fmt.Printf("🎬 SRT Subtitles   : %s\n", srtFile)
	}
}
