package com.localforge.app.engine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Your own model on a rented GPU: a llama.cpp server reached over HTTPS with your key. Same chat format and
 * streaming as on the phone, just a much bigger GPU.
 *
 * Two kinds of address:
 *  - a fixed https:// address (e.g. remote/modal_engine.py on Modal);
 *  - a Colab code: the ntfy topic the LocalForge Colab notebook announces itself on. Colab gets a new
 *    trycloudflare address on every start, so the app looks up the current one there each time it's needed.
 */
object RemoteEngine {
    /** What was typed: an https:// address or a Colab code. */
    @Volatile var source: String = ""
        private set
    /** The address requests go to right now (for a Colab code, the latest one Colab announced). */
    @Volatile var url: String = ""
        private set
    @Volatile private var key: String = ""
    @Volatile var model: String = ""
        private set
    @Volatile var ctx: Int = 0
        private set
    /** Set up and used for chats (the server itself may be asleep; it's woken on the next request). */
    @Volatile var ready = false
    @Volatile private var conn: HttpURLConnection? = null
    @Volatile private var stopped = false
    @Volatile private var lastOk = 0L
    @Volatile var lastTokensPerSecond = 0.0
        private set
    /** What the remote engine is doing right now ("Waking the GPU…"), for the screen. */
    @Volatile var phase: String? = null
        private set

    private val TUNNEL = Regex("^https://[a-z0-9-]+\\.trycloudflare\\.com$")
    private val TOPIC = Regex("^[A-Za-z0-9_-]{6,64}$")

    /** True when the address is a Colab code rather than a fixed https:// address. */
    val isColab: Boolean get() = !source.startsWith("http://") && !source.startsWith("https://")

    /** The server has been quiet long enough that it has probably scaled to zero (or Colab moved). */
    fun asleep(): Boolean = System.currentTimeMillis() - lastOk > 90_000

    fun configure(u: String, k: String, savedModel: String = "", savedCtx: Int = 0) {
        source = u.trim().trimEnd('/').removePrefix("ntfy.sh/").removePrefix("https://ntfy.sh/")
        url = if (isColab) "" else source
        key = k.trim()
        model = savedModel
        ctx = savedCtx
    }

    private fun open(path: String, method: String = "GET", timeoutMs: Int = 30_000): HttpURLConnection =
        (URL(url + path).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 30_000
            readTimeout = timeoutMs
            setRequestProperty("Authorization", "Bearer $key")
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("User-Agent", "LocalForge/1.0")
        }

    private fun readAll(c: HttpURLConnection): String =
        (if (c.responseCode in 200..299) c.inputStream else (c.errorStream ?: c.inputStream)).bufferedReader().use { it.readText() }

    /**
     * For a Colab code: the address Colab last announced as ready, or null when its latest news is that it
     * stopped (or there's no news in the last 12 hours, Colab's hard limit).
     */
    private fun colabAddress(): String? {
        if (!TOPIC.matches(source)) throw IllegalStateException("That isn't an https:// address or a Colab code.")
        val c = URL("https://ntfy.sh/$source/json?poll=1&since=12h").openConnection() as HttpURLConnection
        c.connectTimeout = 20_000; c.readTimeout = 20_000
        c.setRequestProperty("User-Agent", "LocalForge/1.0")
        val text = try { c.inputStream.bufferedReader().use { it.readText() } } finally { c.disconnect() }
        val news = text.lines().filter { it.isNotBlank() }.mapNotNull { runCatching { JSONObject(it) }.getOrNull() }
            .filter { it.optString("event") == "message" }
        for (m in news.asReversed()) {
            val n = runCatching { JSONObject(m.optString("message", "{}")) }.getOrNull() ?: continue
            when (n.optString("stage")) {
                "ready" -> n.optString("url").trim().trimEnd('/').takeIf { TUNNEL.matches(it) }?.let { return it }
                "stopped", "ended", "wrong_gpu" -> return null
            }
        }
        return null
    }

    /** For a Colab code, looks up the current address. Returns false when Colab isn't running. */
    private fun refreshColab(): Boolean {
        if (!isColab) return true
        val u = colabAddress() ?: return false
        if (u != url) { url = u; lastOk = 0L }
        return true
    }

    private const val NOT_RUNNING = "The Colab model isn't running. Open the LocalForge Colab notebook, press Run all, " +
        "and try again once it says ready."

    /**
     * Waits until the server answers /health: a GPU that scaled to zero takes a minute or two to start and load
     * the model. Gives up after [maxMs].
     */
    suspend fun waitReady(maxMs: Long = 6 * 60_000L) = withContext(Dispatchers.IO) {
        val end = System.currentTimeMillis() + maxMs
        var last = ""
        phase = if (isColab) "Finding the Colab model…" else "Waking the remote GPU…"
        try {
            while (System.currentTimeMillis() < end) {
                if (stopped) throw IllegalStateException("Stopped.")
                try {
                    if (isColab && !refreshColab()) throw IllegalStateException(NOT_RUNNING)
                    val c = open("/health", timeoutMs = 60_000)
                    val code = c.responseCode
                    c.disconnect()
                    if (code == 200) { lastOk = System.currentTimeMillis(); return@withContext }
                    last = "HTTP $code"
                    if (code == 401) throw IllegalStateException("Wrong key: the remote engine refused it.")
                    if (code == 404 && !isColab) throw IllegalStateException("Nothing at $url (HTTP 404). Check the address.")
                    if (code == 503) phase = "The remote GPU is loading the model…"
                } catch (e: IOException) {
                    last = e.message ?: e.javaClass.simpleName
                }
                delay(4000)
            }
            throw IllegalStateException(
                if (isColab) "The Colab model didn't answer ($last). Check the Colab tab is still running."
                else "The remote engine didn't wake up ($last). Check it's deployed and has credit left."
            )
        } finally {
            phase = null
        }
    }

    /** Checks the address and key, and reads the model name and context size. */
    suspend fun connect(): String = withContext(Dispatchers.IO) {
        stopped = false
        if (!isColab && !source.startsWith("https://") && !source.startsWith("http://")) {
            throw IllegalStateException("The address must start with https://")
        }
        waitReady()
        val c = open("/v1/models")
        val code = c.responseCode
        val body = readAll(c)
        c.disconnect()
        if (code == 401) throw IllegalStateException("Wrong key: the remote engine refused it.")
        if (code !in 200..299) throw IllegalStateException("The remote engine answered HTTP $code.")
        model = JSONObject(body).optJSONArray("data")?.optJSONObject(0)?.optString("id").orEmpty().ifBlank { "remote model" }
        ctx = try {
            val p = open("/props")
            val o = JSONObject(readAll(p)); p.disconnect()
            o.optJSONObject("default_generation_settings")?.optInt("n_ctx", 0)?.takeIf { it > 0 } ?: o.optInt("n_ctx", 8192)
        } catch (_: Exception) { 8192 }
        ready = true
        model
    }

    /** Forgets an earlier Stop, before a new wake-up. */
    fun clearStop() { stopped = false }

    fun stop() {
        stopped = true
        try { conn?.disconnect() } catch (_: Exception) {}
    }

    /**
     * Tells the Colab notebook someone is still using it, so its idle watchdog doesn't switch it off during a
     * long reply. Sends no idle rule, so whatever LocalForge on the PC chose stays in place.
     */
    private fun colabKeepalive() {
        if (!isColab || url.isBlank()) return
        try {
            val c = open("/keepalive", "POST", timeoutMs = 20_000)
            c.doOutput = true
            c.outputStream.use { it.write("{}".toByteArray()) }
            c.responseCode
            c.disconnect()
        } catch (_: Exception) {}
    }

    /** Streams a reply. [prefill] is the start of the reply that's already written (the model carries on). */
    suspend fun chat(
        messages: List<ChatMessage>, maxTokens: Int, temperature: Float, noThink: Boolean, prefill: String,
        onText: (String) -> Unit
    ): String = withContext(Dispatchers.IO) {
        stopped = false
        // A server that has been quiet for a while has probably scaled to zero (or Colab moved): wake/find it first.
        if (asleep()) waitReady()
        val msgs = JSONArray()
        messages.forEach { msgs.put(JSONObject().put("role", it.role).put("content", it.content)) }
        if (prefill.isNotEmpty()) msgs.put(JSONObject().put("role", "assistant").put("content", prefill))
        val body = JSONObject().put("messages", msgs).put("stream", true).put("max_tokens", maxTokens)
            .put("temperature", temperature.toDouble()).put("cache_prompt", true)
        if (noThink) body.put("chat_template_kwargs", JSONObject().put("enable_thinking", false).put("thinking", false))
        val bytes = body.toString().toByteArray(Charsets.UTF_8)

        // Keep the Colab notebook awake while this reply is being written (big models write slowly).
        val pinger = if (isColab) Thread {
            try { while (true) { colabKeepalive(); Thread.sleep(120_000) } } catch (_: InterruptedException) {}
        }.apply { isDaemon = true; start() } else null

        val out = StringBuilder()
        try {
            // Cloudflare's tunnel gives up (HTTP 524) when a big model needs over ~100 s to read a long prompt
            // before its first word. The server keeps what it has read, so asking again continues from there.
            var tries = 0
            var answer: HttpURLConnection? = null
            while (answer == null) {
                if (stopped) return@withContext out.toString()
                val c = open("/v1/chat/completions", "POST", timeoutMs = 5 * 60_000)
                conn = c
                try {
                    c.doOutput = true
                    c.outputStream.use { it.write(bytes) }
                    val code = c.responseCode
                    tries++
                    if (code == 524 && tries < 15) {
                        phase = "The model is still reading the conversation…"
                        c.disconnect(); continue
                    }
                    if (isColab && (code == 530 || code == 404 || code == 502) && tries < 3) {
                        // The tunnel is gone: Colab restarted on a new address, or stopped.
                        c.disconnect()
                        if (!refreshColab()) throw IllegalStateException(NOT_RUNNING)
                        continue
                    }
                    if (code == 401) throw IllegalStateException("Wrong key: the remote engine refused it.")
                    if (code == 400 && readAll(c).contains("context", ignoreCase = true)) {
                        throw IllegalStateException("The conversation is too long for the remote model's context. Clear the chat.")
                    }
                    if (code !in 200..299) throw IllegalStateException("The remote engine answered HTTP $code.")
                    answer = c
                } catch (e: IOException) {
                    c.disconnect()
                    if (stopped) return@withContext out.toString()
                    tries++
                    if (isColab && tries < 3 && refreshColab()) continue
                    throw IllegalStateException(
                        if (isColab) NOT_RUNNING else "Lost the connection to the remote engine (${e.message})."
                    )
                }
            }
            phase = null
            val c: HttpURLConnection = answer
            try {
                c.inputStream.bufferedReader(Charsets.UTF_8).useLines { lines ->
                    for (raw in lines) {
                        if (stopped) break
                        val line = raw.trim()
                        if (!line.startsWith("data:")) continue
                        val data = line.removePrefix("data:").trim()
                        if (data == "[DONE]") break
                        val o = try { JSONObject(data) } catch (_: Exception) { continue }
                        o.optJSONArray("choices")?.optJSONObject(0)?.optJSONObject("delta")?.let { d ->
                            if (d.has("content") && !d.isNull("content")) {
                                val piece = d.optString("content")
                                if (piece.isNotEmpty()) { out.append(piece); onText(piece) }
                            }
                        }
                        o.optJSONObject("timings")?.let { t -> lastTokensPerSecond = t.optDouble("predicted_per_second", lastTokensPerSecond) }
                    }
                }
                lastOk = System.currentTimeMillis()
            } catch (e: IOException) {
                if (!stopped) throw IllegalStateException("Lost the connection to the remote engine (${e.message}).")
            } finally {
                c.disconnect()
            }
        } finally {
            conn = null
            phase = null
            pinger?.interrupt()
        }
        out.toString()
    }
}
