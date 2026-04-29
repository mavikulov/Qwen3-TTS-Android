package com.example.qwen3_tts.audio

import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import kotlin.math.roundToInt

object WavFileWriter {

    fun writeMono16BitPcm(
        file: File,
        waveform: FloatArray,
        sampleRate: Int = 24000
    ) {
        val numChannels = 1
        val bitsPerSample = 16
        val bytesPerSample = bitsPerSample / 8
        val dataSize = waveform.size * bytesPerSample
        val byteRate = sampleRate * numChannels * bytesPerSample
        val blockAlign = numChannels * bytesPerSample
        val chunkSize = 36 + dataSize

        file.parentFile?.mkdirs()

        DataOutputStream(FileOutputStream(file)).use { out ->
            out.writeBytes("RIFF")
            out.writeIntLE(chunkSize)
            out.writeBytes("WAVE")
            out.writeBytes("fmt ")
            out.writeIntLE(16)
            out.writeShortLE(1)
            out.writeShortLE(numChannels)
            out.writeIntLE(sampleRate)
            out.writeIntLE(byteRate)
            out.writeShortLE(blockAlign)
            out.writeShortLE(bitsPerSample)
            out.writeBytes("data")
            out.writeIntLE(dataSize)

            for (sample in waveform) {
                val clamped = sample.coerceIn(-1f, 1f)
                val pcm = (clamped * 32767f).roundToInt()
                out.writeShortLE(pcm)
            }
        }
    }

    private fun DataOutputStream.writeIntLE(value: Int) {
        writeByte(value and 0xFF)
        writeByte((value ushr 8) and 0xFF)
        writeByte((value ushr 16) and 0xFF)
        writeByte((value ushr 24) and 0xFF)
    }

    private fun DataOutputStream.writeShortLE(value: Int) {
        writeByte(value and 0xFF)
        writeByte((value ushr 8) and 0xFF)
    }
}