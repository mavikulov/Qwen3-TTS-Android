package com.example.qwen3_tts

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.qwen3_tts.storage.BundleImporter
import com.example.qwen3_tts.domain.pipeline.QwenTtsPipeline

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "Qwen3TTS"
    }

    private var mediaPlayer: android.media.MediaPlayer? = null
    private var lastOutputFile: java.io.File? = null

    private lateinit var spSpeaker: Spinner
    private lateinit var etInputText: EditText
    private lateinit var progressBar: ProgressBar
    private lateinit var tvProgressStatus: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvOutputPath: TextView

    private val importZipLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri == null) {
                setStatus("Import cancelled")
                return@registerForActivityResult
            }

            Thread {
                try {
                    runOnUiThread { setStatus("Importing bundle...") }

                    val importer = BundleImporter(this, TAG)
                    importer.importZipFromUri(uri)

                    runOnUiThread {
                        setStatus("Bundle imported successfully")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Bundle import failed", e)
                    runOnUiThread {
                        setStatus("Bundle import failed: ${e.message}")
                    }
                }
            }.start()
        }

    private fun setupSpeakerSpinner() {
        try {
            val pipeline = QwenTtsPipeline(this, TAG)
            val speakers = pipeline.loadAvailableSpeakers()

            val adapter = ArrayAdapter(
                this,
                android.R.layout.simple_spinner_item,
                speakers
            )
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

            spSpeaker.adapter = adapter

            val defaultIndex = speakers.indexOf("ryan")
            if (defaultIndex >= 0) {
                spSpeaker.setSelection(defaultIndex)
            }

            setStatus("Loaded ${speakers.size} speakers")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load speakers", e)
            setStatus("Failed to load speakers: ${e.message}")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        spSpeaker = findViewById(R.id.spSpeaker)
        etInputText = findViewById(R.id.etInputText)
        progressBar = findViewById(R.id.progressBar)
        tvProgressStatus = findViewById(R.id.tvProgressStatus)
        tvStatus = findViewById(R.id.tvStatus)
        tvOutputPath = findViewById(R.id.tvOutputPath)

        setupSpeakerSpinner()
        setProgress(0, "Idle")

        findViewById<Button>(R.id.btnImportBundle).setOnClickListener {
            importZipLauncher.launch(arrayOf("application/zip", "application/octet-stream"))
        }

        findViewById<Button>(R.id.btnGenerate).setOnClickListener {
            generateSpeechFromUi()
        }

        findViewById<Button>(R.id.btnPlay).setOnClickListener {
            playLastOutput()
        }

        findViewById<Button>(R.id.btnStop).setOnClickListener {
            stopPlayback()
        }
    }

    private fun generateSpeechFromUi() {
        val rawInputText = etInputText.text.toString()

        if (rawInputText.isBlank()) {
            setStatus("Please enter text")
            return
        }

        val speakerName = spSpeaker.selectedItem?.toString()

        Thread {
            try {
                runOnUiThread {
                    setProgress(0, "Starting...")
                    setStatus("Generating speech...")
                    tvOutputPath.text = "Output: processing..."
                }

                val pipeline = QwenTtsPipeline(this, TAG)
                val result = pipeline.run(
                    rawInputText = rawInputText,
                    language = "auto",
                    speakerName = speakerName,
                    maxSentencesPerChunk = 2,
                    temperature = 1.0f,
                    topK = 50,
                    repetitionPenalty = 1.1f,
                    minNewTokens = 0,
                    silenceBetweenChunksMs = 250,
                    onProgress = { update ->
                        runOnUiThread {
                            setProgress(update.percent, update.message)
                        }
                    }
                )

                lastOutputFile = result.outputWav

                runOnUiThread {
                    setProgress(100, "Completed")
                    setStatus(
                        "Done. Speaker: ${result.selectedSpeakerName ?: "auto"} (#${result.selectedSpeakerId}), " +
                                "chunks: ${result.chunks.size}, duration: ${"%.2f".format(result.totalDurationSec)} sec"
                    )
                    tvOutputPath.text = "Output: ${result.outputWav.absolutePath}"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Speech generation failed", e)
                runOnUiThread {
                    setProgress(0, "Failed")
                    setStatus("Generation failed: ${e.message}")
                    tvOutputPath.text = "Output: error"
                }
            }
        }.start()
    }

    private fun playLastOutput() {
        val file = lastOutputFile
        if (file == null) {
            Log.e(TAG, "Play requested but lastOutputFile is null")
            setStatus("No output file to play")
            return
        }

        Log.i(TAG, "Play requested for file=${file.absolutePath}")
        Log.i(TAG, "File exists=${file.exists()} size=${if (file.exists()) file.length() else 0L}")

        if (!file.exists() || file.length() <= 44L) {
            setStatus("Output file is missing or too small")
            return
        }

        try {
            releasePlayerOnly()

            mediaPlayer = android.media.MediaPlayer().apply {
                setAudioAttributes(
                    android.media.AudioAttributes.Builder()
                        .setUsage(android.media.AudioAttributes.USAGE_MEDIA)
                        .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )

                setDataSource(file.absolutePath)
                setVolume(1.0f, 1.0f)
                isLooping = false

                setOnPreparedListener { mp ->
                    Log.i(TAG, "MediaPlayer prepared, starting playback")
                    Log.i(TAG, "MediaPlayer duration=${mp.duration} ms")
                    mp.start()
                    Log.i(TAG, "MediaPlayer isPlaying=${mp.isPlaying}")
                    setStatus("Playing output...")
                }

                setOnCompletionListener {
                    Log.i(TAG, "Playback completed")
                    setStatus("Playback finished")
                }

                setOnErrorListener { _, what, extra ->
                    Log.e(TAG, "MediaPlayer error what=$what extra=$extra")
                    setStatus("Playback failed: what=$what extra=$extra")
                    true
                }

                prepareAsync()
            }

            setStatus("Preparing playback...")
        } catch (e: Exception) {
            Log.e(TAG, "Playback failed", e)
            setStatus("Playback failed: ${e.message}")
        }
    }

    private fun stopPlayback() {
        try {
            mediaPlayer?.stop()
        } catch (_: Exception) {
        }

        try {
            mediaPlayer?.release()
        } catch (_: Exception) {
        }

        mediaPlayer = null
        setStatus("Playback stopped")
    }

    private fun releasePlayerOnly() {
        try {
            mediaPlayer?.stop()
        } catch (_: Exception) {
        }

        try {
            mediaPlayer?.release()
        } catch (_: Exception) {
        }

        mediaPlayer = null
    }

    private fun setStatus(text: String) {
        tvStatus.text = "Status: $text"
    }

    override fun onDestroy() {
        super.onDestroy()
        stopPlayback()
    }

    private fun setProgress(percent: Int, message: String) {
        progressBar.progress = percent
        tvProgressStatus.text = "Progress: $message ($percent%)"
    }
}