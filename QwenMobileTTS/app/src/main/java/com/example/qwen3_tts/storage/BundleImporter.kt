package com.example.qwen3_tts.storage

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.util.Log
import java.io.File
import java.io.InputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

class BundleImporter(
    private val context: Context,
    private val tag: String = "Qwen3TTS"
) {
    private val paths = QwenStoragePaths(context)

    init {
        paths.ensureDirectories()
    }

    fun importZipFromUri(uri: Uri) {
        val resolver: ContentResolver = context.contentResolver

        resolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Failed to open input stream for URI: $uri" }
            unzipToRuntimeDir(input)
        }
    }

    private fun unzipToRuntimeDir(inputStream: InputStream) {
        ZipInputStream(inputStream).use { zip ->
            var entry: ZipEntry? = zip.nextEntry

            while (entry != null) {
                val entryName = entry.name.replace("\\", "/")
                Log.i(tag, "ZIP entry: $entryName")

                if (!entry.isDirectory) {
                    val outFile = resolveOutputFile(entryName)
                    if (outFile != null) {
                        outFile.parentFile?.mkdirs()
                        outFile.outputStream().use { output ->
                            zip.copyTo(output)
                        }
                        Log.i(tag, "Extracted: ${outFile.absolutePath} size=${outFile.length()}")
                    } else {
                        Log.i(tag, "Skipped entry outside expected structure: $entryName")
                    }
                }

                zip.closeEntry()
                entry = zip.nextEntry
            }
        }
    }

    private fun resolveOutputFile(entryName: String): File? {
        val normalized = entryName.trimStart('/')

        return when {
            normalized.startsWith("models/") -> {
                val relative = normalized.removePrefix("models/")
                File(paths.modelsDir, relative)
            }
            normalized.startsWith("embeddings/") -> {
                val relative = normalized.removePrefix("embeddings/")
                File(paths.embeddingsDir, relative)
            }
            normalized.startsWith("tokenizer/") -> {
                val relative = normalized.removePrefix("tokenizer/")
                File(paths.tokenizerDir, relative)
            }
            else -> null
        }
    }
}