package com.example.qwen3_tts.tokenizer

class QwenByteEncoder {
    val byteToUnicode: Map<Int, String> = buildByteToUnicode()

    fun encodeToByteLevelPieces(text: String): String {
        val bytes = text.toByteArray(Charsets.UTF_8)
        val sb = StringBuilder()
        for (b in bytes) {
            val unsigned = b.toInt() and 0xFF
            sb.append(byteToUnicode[unsigned] ?: error("Missing byte mapping for $unsigned"))
        }
        return sb.toString()
    }

    private fun buildByteToUnicode(): Map<Int, String> {
        val bs = mutableListOf<Int>()

        for (i in 33..126) bs.add(i)
        for (i in 161..172) bs.add(i)
        for (i in 174..255) bs.add(i)

        val cs = bs.toMutableList()
        var n = 0
        for (b in 0..255) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }

        return bs.zip(cs).associate { (b, c) -> b to c.toChar().toString() }
    }
}