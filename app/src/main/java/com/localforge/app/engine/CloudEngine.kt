package com.localforge.app.engine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Cloud aggregators (OpenRouter, Featherless, NanoGPT, Chutes or any OpenAI-style service) with your own key.
 * Only models whose name or description says uncensored or abliterated are offered, biggest first.
 */
object CloudEngine {
    data class Provider(val name: String, val base: String, val keyHint: String, val keyUrl: String)

    val PROVIDERS = listOf(
        Provider("Hugging Face", "https://router.huggingface.co/v1", "hf_…", "https://huggingface.co/settings/tokens"),
        Provider("OpenRouter", "https://openrouter.ai/api/v1", "sk-or-…", "https://openrouter.ai/settings/keys"),
        Provider("Featherless", "https://api.featherless.ai/v1", "rc_…", "https://featherless.ai/account/api-keys"),
        Provider("NanoGPT", "https://nano-gpt.com/api/v1", "key", "https://nano-gpt.com/api"),
        Provider("Chutes", "https://llm.chutes.ai/v1", "cpk_…", "https://chutes.ai/app/api"),
    )

    data class CloudModel(val id: String, val name: String, val ctx: Int, val free: Boolean, val about: String) {
        val short: String get() = name.ifBlank { id }.substringAfter(": ").substringAfterLast('/')
        /** Size in billions of parameters from the name (0 if it doesn't say). */
        val billions: Double get() = Regex("(?<![a-z0-9.])(\\d+(?:\\.\\d+)?)b(?![a-z])", RegexOption.IGNORE_CASE)
            .findAll("$id $name").mapNotNull { it.groupValues[1].toDoubleOrNull() }.maxOrNull() ?: 0.0
    }

    /**
     * Abliterated 70B+ models Hugging Face serves through Featherless. A free Hugging Face account's monthly
     * credit pays for them (a small amount), so they're the only big abliterated models usable for free.
     */
    private val HF_KNOWN = listOf(
        "huihui-ai/Qwen2.5-72B-Instruct-abliterated", "zetasepic/Qwen2.5-72B-Instruct-abliterated-v2",
        "huihui-ai/Llama-3.3-70B-Instruct-abliterated", "huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated",
        "failspy/Meta-Llama-3-70B-Instruct-abliterated-v3.5", "Guilherme34/Hermes-3-Llama-3.1-70B-Uncensored",
    )

    /** Harsh's rule: the name or description must say uncensored or abliterated ("heretic" alone doesn't count). */
    private val UNCENSORED = Regex("uncensor|abliterat", RegexOption.IGNORE_CASE)

    @Volatile var base: String = PROVIDERS[0].base
        private set
    @Volatile private var key: String = ""
    @Volatile var model: String = ""
        private set
    @Volatile var ctx: Int = 8192
        private set
    @Volatile var ready = false
    @Volatile var lastTokensPerSecond = 0.0
        private set
    @Volatile private var stopped = false
    @Volatile private var conn: HttpURLConnection? = null

    val providerName: String get() = PROVIDERS.firstOrNull { it.base == base }?.name ?: base.removePrefix("https://").substringBefore('/')

    fun configure(b: String, k: String, m: String = "", c: Int = 8192) {
        base = b.trim().trimEnd('/'); key = k.trim(); model = m; ctx = c
    }

    private fun open(path: String, method: String = "GET", readMs: Int = 60_000): HttpURLConnection =
        (URL(base + path).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 30_000
            readTimeout = readMs
            if (key.isNotBlank()) setRequestProperty("Authorization", "Bearer $key")
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("User-Agent", "LocalForge/1.0")
            setRequestProperty("HTTP-Referer", "https://github.com/Xwnin/race")
            setRequestProperty("X-Title", "LocalForge Cloud")
        }

    private fun readAll(c: HttpURLConnection): String =
        (if (c.responseCode in 200..299) c.inputStream else (c.errorStream ?: c.inputStream)).bufferedReader().use { it.readText() }

    private fun errorText(code: Int, body: String): String {
        val msg = try { JSONObject(body).optJSONObject("error")?.optString("message").orEmpty() } catch (_: Exception) { "" }
            .ifBlank { body.take(200) }
        return when (code) {
            401, 403 -> "$providerName refused the key. Check it in the Cloud card."
            402 -> "$providerName says this model needs credit. Pick one marked FREE."
            429 -> "$providerName's free limit is used up for now ($msg). Wait a bit or pick another free model."
            else -> "$providerName answered HTTP $code: $msg"
        }
    }

    /** The provider's models that say uncensored or abliterated: free ones first, then biggest first. */
    suspend fun listModels(): List<CloudModel> = withContext(Dispatchers.IO) {
        if (base.contains("router.huggingface.co")) {
            if (key.isBlank()) throw IllegalStateException("Paste your Hugging Face token (hf_…) first.")
            return@withContext HF_KNOWN.map { CloudModel("$it:featherless-ai", it.substringAfter('/'), 32768, true, "abliterated, via Featherless") }
        }
        val c = open("/models")
        val code = c.responseCode
        val body = readAll(c); c.disconnect()
        if (code !in 200..299) throw IllegalStateException(errorText(code, body))
        val arr = try { JSONObject(body).optJSONArray("data") } catch (_: Exception) { null } ?: JSONArray(body)
        (0 until arr.length()).mapNotNull { arr.optJSONObject(it) }.map { o ->
            val id = o.optString("id")
            val p = o.optJSONObject("pricing")
            val free = id.endsWith(":free") || (p != null && p.optString("prompt") in setOf("0", "0.0") && p.optString("completion") in setOf("0", "0.0"))
            CloudModel(id, o.optString("name"), o.optInt("context_length", o.optInt("context_size", o.optInt("max_model_len", 8192))).takeIf { it > 0 } ?: 8192,
                free, o.optString("description").take(400))
        }.filter { UNCENSORED.containsMatchIn(it.id + " " + it.name + " " + it.about) }
            .sortedWith(compareByDescending<CloudModel> { it.free }.thenByDescending { it.billions })
    }

    fun use(m: CloudModel) {
        model = m.id; ctx = m.ctx.coerceIn(4096, 131072); ready = true
    }

    fun stop() {
        stopped = true
        try { conn?.disconnect() } catch (_: Exception) {}
    }

    /** Streams a reply in the OpenAI format. [prefill] is the start of the reply that's already written. */
    suspend fun chat(
        messages: List<ChatMessage>, maxTokens: Int, temperature: Float, prefill: String, onText: (String) -> Unit
    ): String = withContext(Dispatchers.IO) {
        stopped = false
        val msgs = JSONArray()
        messages.forEach { msgs.put(JSONObject().put("role", it.role).put("content", it.content)) }
        if (prefill.isNotEmpty()) msgs.put(JSONObject().put("role", "assistant").put("content", prefill))
        val body = JSONObject().put("model", model).put("messages", msgs).put("stream", true)
            .put("max_tokens", maxTokens).put("temperature", temperature.toDouble())
        val c = open("/chat/completions", "POST", readMs = 5 * 60_000)
        conn = c
        val out = StringBuilder()
        val t0 = System.nanoTime()
        try {
            c.doOutput = true
            c.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
            val code = c.responseCode
            if (code !in 200..299) throw IllegalStateException(errorText(code, readAll(c)))
            c.inputStream.bufferedReader(Charsets.UTF_8).useLines { lines ->
                for (raw in lines) {
                    if (stopped) break
                    val line = raw.trim()
                    if (!line.startsWith("data:")) continue
                    val data = line.removePrefix("data:").trim()
                    if (data == "[DONE]") break
                    val o = try { JSONObject(data) } catch (_: Exception) { continue }
                    o.optJSONObject("error")?.let { e -> throw IllegalStateException("$providerName: ${e.optString("message").take(300)}") }
                    val d = o.optJSONArray("choices")?.optJSONObject(0)?.optJSONObject("delta") ?: continue
                    if (d.has("content") && !d.isNull("content")) {
                        val piece = d.optString("content")
                        if (piece.isNotEmpty()) { out.append(piece); onText(piece) }
                    }
                }
            }
        } catch (e: IOException) {
            if (!stopped) throw IllegalStateException("Lost the connection to $providerName (${e.message}).")
        } finally {
            conn = null
            c.disconnect()
        }
        val secs = (System.nanoTime() - t0) / 1e9
        lastTokensPerSecond = if (secs > 0) (out.length / 3.5) / secs else 0.0
        if (out.isEmpty() && !stopped) throw IllegalStateException("$model sent back an empty reply. Try again or pick another model.")
        out.toString()
    }
}
