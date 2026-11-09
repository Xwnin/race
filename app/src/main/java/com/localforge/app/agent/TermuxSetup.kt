package com.localforge.app.agent

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * "Set up Termux for me": downloads the official Termux APK from GitHub and opens Android's installer, asks for the
 * RUN_COMMAND permission, and then installs every package LocalForge uses (compilers, Node, git, REA's needs)
 * through Termux itself. The one thing an app can't do for you is switch on Termux's "allow-external-apps":
 * only Termux can write its own settings, so you paste one short line into Termux once.
 */
object TermuxSetup {
    /** The one line you paste into Termux: storage access, let LocalForge send it commands, reload. */
    const val PASTE_COMMAND =
        "termux-setup-storage; mkdir -p ~/.termux && (grep -q '^allow-external-apps=true' ~/.termux/termux.properties 2>/dev/null || " +
            "echo 'allow-external-apps=true' >> ~/.termux/termux.properties) && termux-reload-settings && echo 'Done. Go back to LocalForge.'"

    /** Everything LocalForge's runners, <install> and REA use. */
    const val PACKAGES = "clang golang rust openjdk-21 kotlin ruby php nodejs-lts lua54 perl git python make procps binutils"

    private const val RELEASES = "https://api.github.com/repos/termux/termux-app/releases/latest"
    private const val PACKAGES_TIMEOUT_MS = 45 * 60_000L

    private fun apkFile(ctx: Context) = File(ctx.filesDir, "termux/termux.apk")

    /** Downloads the newest Termux APK for this phone's CPU. [progress] gets 0..100. */
    suspend fun download(ctx: Context, progress: (Int) -> Unit): File = withContext(Dispatchers.IO) {
        val rel = get(RELEASES)
        val assets = JSONObject(rel).getJSONArray("assets")
        val names = (0 until assets.length()).map { assets.getJSONObject(it) }.filter { it.optString("name").endsWith(".apk") }
        val abi = Build.SUPPORTED_ABIS.firstOrNull() ?: "arm64-v8a"
        val pick = names.firstOrNull { it.optString("name").contains(abi) }
            ?: names.firstOrNull { it.optString("name").contains("universal") }
            ?: names.firstOrNull()
            ?: error("The Termux release has no APK")
        val out = apkFile(ctx).also { it.parentFile?.mkdirs() }
        val tmp = File(out.parentFile, "termux.apk.part")
        val c = URL(pick.getString("browser_download_url")).openConnection() as HttpURLConnection
        c.connectTimeout = 20_000
        c.readTimeout = 60_000
        c.instanceFollowRedirects = true
        try {
            if (c.responseCode !in 200..299) error("GitHub answered ${c.responseCode}")
            val total = c.contentLengthLong.takeIf { it > 0 } ?: pick.optLong("size", -1)
            var done = 0L
            var last = -1
            c.inputStream.use { input ->
                tmp.outputStream().use { o ->
                    val buf = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        o.write(buf, 0, n)
                        done += n
                        if (total > 0) {
                            val pct = (done * 100 / total).toInt()
                            if (pct != last) { last = pct; withContext(Dispatchers.Main) { progress(pct) } }
                        }
                    }
                }
            }
        } finally {
            c.disconnect()
        }
        out.delete()
        if (!tmp.renameTo(out)) error("Couldn't save the APK")
        out
    }

    private fun get(url: String): String {
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = 15_000
        c.readTimeout = 30_000
        c.setRequestProperty("Accept", "application/vnd.github+json")
        c.setRequestProperty("User-Agent", "LocalForge-Android")
        return try {
            if (c.responseCode !in 200..299) error("GitHub answered ${c.responseCode}")
            c.inputStream.bufferedReader().use { it.readText() }
        } finally {
            c.disconnect()
        }
    }

    /** Android only lets an app open the installer once "Install unknown apps" is on for it. */
    fun canInstall(ctx: Context): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.O || ctx.packageManager.canRequestPackageInstalls()

    fun unknownAppsSettings(ctx: Context): Intent =
        Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:${ctx.packageName}"))

    /** Opens Android's "Install Termux?" screen for the downloaded APK. */
    fun installIntent(ctx: Context, apk: File): Intent {
        val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", apk)
        return Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "application/vnd.android.package-archive")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
        }
    }

    fun openTermux(ctx: Context): Intent? =
        ctx.packageManager.getLaunchIntentForPackage(TermuxBridge.PACKAGE)?.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

    /** Whether Termux accepts LocalForge's commands yet (it doesn't until the pasted line ran). */
    suspend fun answers(ctx: Context): Pair<Boolean, String> {
        val r = TermuxBridge.run(ctx, "echo LF_OK", "/data/data/com.termux/files/home", 20_000)
            ?: return false to "Termux didn't answer. Open Termux once, paste the line, then try again."
        if (r.error != null) return false to "Termux said: ${r.error}\nPaste the line into Termux and press Enter, then come back."
        return (r.stdout.contains("LF_OK")) to r.stdout.trim()
    }

    /** Installs [PACKAGES] through Termux. Returns (ok, last lines of output). */
    suspend fun installPackages(ctx: Context): Pair<Boolean, String> {
        val script = "export DEBIAN_FRONTEND=noninteractive; (yes | pkg update -y) 2>&1 | tail -3; " +
            "(yes | pkg install -y $PACKAGES) 2>&1 | tail -12; " +
            "for t in clang go rustc java node git python; do command -v \$t >/dev/null && echo \"✓ \$t\" || echo \"✗ \$t\"; done; echo SETUP_DONE"
        val r = TermuxBridge.run(ctx, script, "/data/data/com.termux/files/home", PACKAGES_TIMEOUT_MS)
            ?: return false to "Still installing after ${PACKAGES_TIMEOUT_MS / 60_000} minutes. Leave it, then press Finish setup again."
        if (r.error != null) return false to "Termux refused: ${r.error}"
        val out = (r.stdout + "\n" + r.stderr).trim()
        val ok = out.contains("SETUP_DONE") && !out.lines().any { it.startsWith("✗ ") }
        return ok to out.lines().takeLast(14).joinToString("\n")
    }
}
