package com.localforge.app.agent

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * GitHub repos as tools: the AI reads any public repo (<github>), installs it on the phone through Termux
 * (<install>), keeps what it learned about using it (<learn>), and finds it listed in every chat afterwards.
 * REA (morluto/rea) comes with a built-in install recipe and guide, see [RepoLogic.REA].
 */
object GitHubTools {
    private const val API = "https://api.github.com/repos/"
    private const val MAX_TEXT = 9000
    private const val INSTALL_TIMEOUT_MS = 25 * 60_000L

    /** One installed repo, saved so later chats know about it. */
    data class Installed(val repo: String, val dir: String, val commands: String, val notes: String, val time: Long)

    private var file: File? = null
    private val installed = mutableListOf<Installed>()

    @Synchronized
    fun init(ctx: Context) {
        if (file != null) return
        file = File(ctx.applicationContext.filesDir, "github-tools.json")
        load()
    }

    fun list(): List<Installed> = synchronized(this) { installed.toList() }

    // ---------- reading a repo ----------

    private fun get(url: String, accept: String): Pair<Int, String> {
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = 15_000
        c.readTimeout = 30_000
        c.setRequestProperty("Accept", accept)
        c.setRequestProperty("User-Agent", "LocalForge-Android")
        c.setRequestProperty("X-GitHub-Api-Version", "2022-11-28")
        return try {
            val code = c.responseCode
            val stream = if (code in 200..299) c.inputStream else c.errorStream
            code to (stream?.bufferedReader()?.use { it.readText() } ?: "")
        } finally {
            c.disconnect()
        }
    }

    private fun cut(s: String, max: Int = MAX_TEXT) = if (s.length > max) s.take(max) + "\n…[cut, ${s.length - max} more chars]" else s

    private fun encodePath(p: String) = p.trim('/').split('/').joinToString("/") { URLEncoder.encode(it, "UTF-8").replace("+", "%20") }

    private fun failure(repo: String, code: Int, body: String): String = when (code) {
        404 -> "GitHub has no public repo or file at $repo. Check the name (owner/name)."
        403, 429 -> "GitHub's rate limit was hit (60 lookups an hour without an account). Try again later. ${body.take(200)}"
        else -> "GitHub answered $code for $repo: ${body.take(300)}"
    }

    /** Description, README and top-level files of [repo], or one file/folder in it when [path] is set. Blocking. */
    fun read(repoInput: String, path: String = ""): String {
        val repo = RepoLogic.parseRepo(repoInput) ?: return "\"$repoInput\" isn't a GitHub repo. Write it as owner/name, e.g. <github repo=\"morluto/rea\"/>."
        return try {
            if (path.isNotBlank()) readPath(repo, path) else readRepo(repo)
        } catch (e: Exception) {
            "Couldn't reach GitHub for $repo: ${e.message}"
        }
    }

    private fun readRepo(repo: String): String {
        val (code, body) = get(API + repo, "application/vnd.github+json")
        if (code != 200) return failure(repo, code, body)
        val o = JSONObject(body)
        val sb = StringBuilder("GitHub repo ${o.optString("full_name", repo)}")
        o.optString("description").takeIf { it.isNotBlank() && it != "null" }?.let { sb.append(": ").append(it) }
        sb.append("\nLanguage: ${o.optString("language", "?")} · ★ ${o.optInt("stargazers_count")} · default branch ${o.optString("default_branch")}")
        o.optJSONObject("license")?.optString("spdx_id")?.let { sb.append(" · license $it") }
        val (lc, listing) = get(API + repo + "/contents", "application/vnd.github+json")
        if (lc == 200) {
            val a = JSONArray(listing)
            val names = (0 until a.length()).map { a.getJSONObject(it) }.map { if (it.optString("type") == "dir") it.optString("name") + "/" else it.optString("name") }
            sb.append("\nFiles: ").append(names.take(80).joinToString(", "))
        }
        val (rc, readme) = get(API + repo + "/readme", "application/vnd.github.raw")
        sb.append("\n\nREADME:\n").append(if (rc == 200) cut(readme, 7000) else "(no README)")
        installedEntry(repo)?.let { sb.append("\n\n[Already installed on this phone in ${it.dir}. Commands: ${it.commands.ifBlank { "see README" }}]") }
            ?: sb.append("\n\n[To install it on this phone: <install repo=\"$repo\"/>]")
        RepoLogic.knownFor(repo)?.let { sb.append("\n\n[LocalForge's own guide for ${it.name}]\n${it.guide}") }
        return sb.toString()
    }

    private fun readPath(repo: String, path: String): String {
        val (code, body) = get(API + repo + "/contents/" + encodePath(path), "application/vnd.github.raw")
        if (code != 200) return failure("$repo/$path", code, body)
        val t = body.trimStart()
        if (t.startsWith("[")) {
            // A folder: the raw media type still lists it as JSON.
            val a = JSONArray(t)
            val names = (0 until a.length()).map { a.getJSONObject(it) }.map { if (it.optString("type") == "dir") it.optString("name") + "/" else it.optString("name") }
            return "Folder $path in $repo:\n" + names.joinToString("\n")
        }
        return "File $path in $repo:\n" + cut(body)
    }

    // ---------- installing ----------

    /** Installs [repoInput] inside Termux and remembers it. Returns the report for the model. */
    suspend fun install(ctx: Context, repoInput: String): Pair<Boolean, String> {
        val repo = RepoLogic.parseRepo(repoInput) ?: return false to "\"$repoInput\" isn't a GitHub repo (write owner/name)."
        if (!TermuxBridge.isInstalled(ctx)) return false to "Installing repos needs Termux, which isn't installed. The user can set it up on the Home tab (Code runners), then ask again."
        if (!TermuxBridge.hasPermission(ctx)) return false to "LocalForge isn't allowed to use Termux yet. The user taps \"Allow\" on the Home tab → Code runners, then asks again."
        val home = "/data/data/com.termux/files/home"
        val r = TermuxBridge.run(ctx, RepoLogic.fullScript(repo), home, INSTALL_TIMEOUT_MS)
            ?: return false to "Termux didn't finish installing $repo within ${INSTALL_TIMEOUT_MS / 60_000} minutes. It may still be going in Termux; check again with <install repo=\"$repo\"/>."
        if (r.error != null) return false to "Termux refused to run the install: ${r.error}\nThe user must run the Termux setup command from the Home tab once (it enables allow-external-apps)."
        val out = (r.stdout + if (r.stderr.isNotBlank()) "\n" + r.stderr else "").trim()
        val shown = if (out.length > 6000) out.take(1500) + "\n…\n" + out.takeLast(4500) else out
        val ok = r.exitCode == 0 && RepoLogic.succeeded(out)
        if (!ok) return false to "Installing $repo failed (exit ${r.exitCode}):\n```\n$shown\n```\nRead the error, fix it with <run lang=\"bash\"> (e.g. pkg install the missing package), then try <install> again."
        val known = RepoLogic.knownFor(repo)
        val commands = RepoLogic.commandsFrom(out)
        val dir = if (known != null && !known.clone) "(installed globally with npm)" else "$home/lf-repos/${RepoLogic.folderOf(repo)}"
        val notes = known?.guide ?: try {
            val (rc, readme) = get(API + repo + "/readme", "application/vnd.github.raw")
            if (rc == 200) RepoLogic.usageOf(readme) else ""
        } catch (_: Exception) { "" }
        remember(Installed(repo, dir, commands, existingNotes(repo) ?: notes, System.currentTimeMillis()))
        val how = known?.guide ?: ("How to use it (from its README):\n" + notes.ifBlank { "(no README; look at its files with <github repo=\"$repo\" file=\"...\"/>)" })
        return true to "Installed $repo on the phone. Folder: $dir. Commands: ${commands.ifBlank { "none found" }}.\n```\n$shown\n```\n$how\n\n" +
            "Now use it with <run lang=\"bash\">, starting with its --help. Once you know how it works, save what you learned with " +
            "<learn repo=\"$repo\">short how-to</learn> so you remember it in later chats."
    }

    // ---------- what the AI learned ----------

    /** Saves the AI's own notes on how to use [repoInput]. */
    fun learn(repoInput: String, notes: String): String {
        val repo = RepoLogic.parseRepo(repoInput) ?: return "\"$repoInput\" isn't a GitHub repo (write owner/name)."
        val clean = notes.trim().take(1500)
        if (clean.isBlank()) return "Nothing to save: write the notes between <learn repo=\"$repo\"> and </learn>."
        val old = installedEntry(repo)
        remember(old?.copy(notes = clean) ?: Installed(repo, "(not installed yet)", "", clean, System.currentTimeMillis()))
        return "Saved how to use $repo. You'll see it in every chat from now on."
    }

    fun forget(repo: String) = synchronized(this) {
        installed.removeAll { it.repo.equals(repo, ignoreCase = true) }
        save()
    }

    private fun installedEntry(repo: String) = synchronized(this) { installed.firstOrNull { it.repo.equals(repo, ignoreCase = true) } }

    /** Notes the AI wrote itself with <learn> win over README text on a reinstall. */
    private fun existingNotes(repo: String): String? = installedEntry(repo)?.notes?.takeIf { RepoLogic.knownFor(repo) == null && it.isNotBlank() }

    private fun remember(e: Installed) = synchronized(this) {
        installed.removeAll { it.repo.equals(e.repo, ignoreCase = true) }
        installed += e
        save()
    }

    // ---------- system prompt ----------

    /** The GitHub part of the system prompt: the tags, the installed tools and REA. */
    fun prompt(): String {
        val sb = StringBuilder(
            """
GITHUB REPOS AS TOOLS: you can learn any public GitHub repo and install it on this phone, then use it yourself.
<github repo="owner/name"/>                     read the repo: description, files and README
<github repo="owner/name" file="docs/cli.md"/>  read one file or folder in it
<install repo="owner/name"/>                    download and install it in Termux (asks the user); you get how to run it
<learn repo="owner/name">how to use it</learn>  save your own short how-to so you remember it in every later chat
When the user links or names a repo and wants to use it: <github> to learn it, <install> it, run its --help with
<run lang="bash">, do the task, then <learn> what worked. Installed commands run from <run lang="bash">.
""".trim()
        )
        val list = list()
        if (list.isNotEmpty()) {
            sb.append("\nInstalled on this phone:")
            for (t in list) {
                sb.append("\n- ${t.repo}: commands ${t.commands.ifBlank { "?" }}, in ${t.dir}")
                if (RepoLogic.knownFor(t.repo) == null && t.notes.isNotBlank()) sb.append("\n  How to use: ").append(t.notes.take(600).replace("\n", "\n  "))
            }
        }
        val rea = RepoLogic.REA
        val reaInstalled = list.any { it.repo.equals(rea.repo, ignoreCase = true) && it.dir != "(not installed yet)" }
        sb.append("\n\n")
        if (!reaInstalled) sb.append("REA is not installed yet. When the user wants to reverse engineer or inspect an app, APK, binary or file, install it first with <install repo=\"${rea.repo}\"/>.\n")
        sb.append(rea.guide)
        return sb.toString()
    }

    // ---------- saving ----------

    private fun load() {
        try {
            val f = file ?: return
            if (!f.exists()) return
            val a = JSONArray(f.readText())
            for (i in 0 until a.length()) a.getJSONObject(i).let {
                installed += Installed(it.optString("repo"), it.optString("dir"), it.optString("commands"), it.optString("notes"), it.optLong("time"))
            }
        } catch (_: Exception) {}
    }

    private fun save() {
        try {
            val f = file ?: return
            val a = JSONArray()
            installed.forEach { a.put(JSONObject().put("repo", it.repo).put("dir", it.dir).put("commands", it.commands).put("notes", it.notes).put("time", it.time)) }
            val tmp = File(f.parentFile, f.name + ".tmp")
            tmp.writeText(a.toString(1))
            if (!tmp.renameTo(f)) { f.delete(); tmp.renameTo(f) }
        } catch (_: Exception) {}
    }
}
