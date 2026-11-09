package com.localforge.app.agent

/**
 * The plain logic behind GitHub repos as tools (no Android, so it is unit tested): reading a repo name the way the
 * model or the user writes it, the bash script that installs a repo inside Termux, and the built-in guides for tools
 * LocalForge already knows, such as REA.
 */
object RepoLogic {
    /** Where Termux keeps installed repos (Termux's own home, where code is allowed to run). */
    const val BASE = "\$HOME/lf-repos"

    private val repoName = Regex("""^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}$""")

    /**
     * "owner/name" from anything the model writes: owner/name, github.com/owner/name, a full https URL,
     * a .git clone URL or a link to a file inside the repo. Null if it isn't a GitHub repo.
     */
    fun parseRepo(input: String): String? {
        var s = input.trim().trim('"', '\'', '<', '>', '(', ')', '.', ',').removeSuffix("/")
        s = s.replace(Regex("^(?:https?://)?(?:www\\.)?github\\.com/", RegexOption.IGNORE_CASE), "")
        s = s.replace(Regex("^git@github\\.com:", RegexOption.IGNORE_CASE), "")
        val parts = s.split('/', '?', '#').filter { it.isNotBlank() }
        if (parts.size < 2) return null
        val r = parts[0] + "/" + parts[1].removeSuffix(".git")
        return if (repoName.matches(r) && !r.endsWith("/.") && !r.endsWith("/..")) r else null
    }

    /** The folder name a repo is cloned into. */
    fun folderOf(repo: String): String = repo.substringAfter('/')

    /** A tool LocalForge already knows: how to install it and how to use it. */
    data class Known(
        val repo: String,
        val name: String,
        /** Shell lines run after the optional clone (in the clone, or in $BASE when [clone] is false). */
        val install: String,
        val clone: Boolean,
        /** Command(s) the tool gives you, for the system prompt. */
        val commands: String,
        /** How to use it, written for the model. */
        val guide: String
    )

    private const val NODE_OK = """node -e 'const v=process.versions.node.split(".").map(Number),[a,b,c]=v;const ok=(a==22&&(b>23||(b==23&&c>=4)))||(a==24&&(b>22||(b==22&&c>=1)))||(a==26&&(b>12||(b==12&&c>=1)))||a>26;process.exit(ok?0:1)' 2>/dev/null"""

    val REA = Known(
        repo = "morluto/rea",
        name = "REA (Reverse Engineer Anything)",
        clone = false,
        commands = "rea",
        install = """
pkgi git procps python make clang binutils
command -v node >/dev/null 2>&1 || pkgi nodejs-lts
if ! $NODE_OK; then
  say "REA needs Node.js 22.23.4+, 24.22.1+ or 26.12.1+. This Node is $(node -v 2>/dev/null). Trying the other Termux Node package."
  pkgi nodejs-lts
  $NODE_OK || pkgi nodejs
fi
$NODE_OK || say "WARNING: Node $(node -v 2>/dev/null) is older than REA wants. REA may refuse to start."
say "npm install -g rea-agents@latest (a few minutes the first time)"
npm install -g --no-audit --no-fund rea-agents@latest 2>&1 | tail -15
command -v rea >/dev/null 2>&1 || { echo "INSTALL_FAILED: the rea command was not installed (see the npm output above)"; exit 3; }
mkdir -p /storage/emulated/0/LocalForge/rea 2>/dev/null
say "Installed: $(rea --version 2>&1 | head -1)"
rea --help 2>&1 | head -12
""".trim(),
        guide = """
REA (Reverse Engineer Anything, github.com/morluto/rea) inspects apps, APKs, binaries, packages and files WITHOUT their
source code and returns JSON "Evidence". Run it with <run lang="bash">. Always give absolute paths. Output is big, so
save it to a file and read the parts you need with python:
<run lang="bash">
rea inspect-artifact /storage/emulated/0/Download/app.apk --json > /storage/emulated/0/LocalForge/rea/app.json; echo saved
</run>
Start with: rea inspect-artifact <file> --json (works on almost anything; says what it is and which commands to try next).
Help: rea --help, rea <command> --help. Useful commands:
- APKs: inspect-apk-signature, identify-flutter-build, inspect-android-package / inspect-android-class /
  inspect-android-method (these three need JADX + Java, may not work on the phone), decode-android-resources (apktool)
- JavaScript / Electron apps or folders: analyze-javascript-application <dir> --json
- Websites: save the browser's traffic with <network save=""/>, then inspect-web-network-capture <file.har> --json;
  environment-build <file.har> <out-dir> --name x --json turns it into an agent environment
- Files: inspect-model-artifact (.gguf/.safetensors), inspect-sqlite-database, inspect-python-pyc, inspect-go-binary,
  inspect-rust-binary, inspect-elf-notes, inspect-binary-layout, inspect-native-exports, inspect-pe-imports,
  inspect-managed-members (.NET), inspect-il2cpp-metadata (Unity), extract-artifact, inspect-byte-entropy
- Native decompiling (decompile, function) needs Ghidra/Hopper/IDA, which don't run on a phone: say so.
Explain what the evidence shows, quote the fields you used, and say what stays unknown. Never invent findings.
""".trim()
    )

    val known: List<Known> = listOf(REA)

    fun knownFor(repo: String): Known? = known.firstOrNull { it.repo.equals(repo, ignoreCase = true) }

    /** Quotes [s] for bash. */
    fun q(s: String) = "'" + s.replace("'", "'\\''") + "'"

    /**
     * The bash script that installs [repo] inside Termux. It clones (or updates) the repo into ~/lf-repos/<name>,
     * then either runs the known recipe or detects the project type (npm, pip, cargo, go, make). It prints
     * "== " progress lines, "COMMANDS: ..." with what it installed, and exits non-zero on failure.
     */
    fun installScript(repo: String): String {
        val k = knownFor(repo)
        val dir = "$BASE/${folderOf(repo)}"
        val clone = k == null || k.clone
        return buildString {
            appendLine("export DEBIAN_FRONTEND=noninteractive")
            appendLine("say(){ echo \"== $*\"; }")
            appendLine("pkgi(){ miss=''; for p in \"$@\"; do dpkg -s \"\$p\" >/dev/null 2>&1 || miss=\"\$miss \$p\"; done; [ -z \"\$miss\" ] && return 0; say \"pkg install\$miss\"; yes | pkg install -y \$miss 2>&1 | tail -4; }")
            appendLine("mkdir -p \"$BASE\" && cd \"$BASE\" || exit 1")
            if (clone) {
                appendLine("command -v git >/dev/null 2>&1 || pkgi git")
                appendLine("DIR=\"$dir\"")
                appendLine("if [ -d \"\$DIR/.git\" ]; then say \"Updating \$DIR\"; git -C \"\$DIR\" pull --ff-only 2>&1 | tail -3")
                appendLine("else say \"Cloning https://github.com/$repo into \$DIR\"; git clone --depth 1 ${q("https://github.com/$repo")} \"\$DIR\" 2>&1 | tail -3")
                appendLine("  [ -d \"\$DIR/.git\" ] || { echo \"CLONE_FAILED: is https://github.com/$repo a public repo?\"; exit 2; }; fi")
                appendLine("cd \"\$DIR\" || exit 2")
            }
            if (k != null) {
                appendLine(k.install)
                appendLine("echo \"COMMANDS: ${k.commands}\"")
            } else {
                append(GENERIC)
            }
            if (clone) appendLine("say \"Files in \$DIR:\"; ls -1 | head -30")
            appendLine("echo INSTALL_OK")
        }
    }

    /** Works out what kind of project it is and installs it the usual way. */
    private val GENERIC = """
CMDS=''
DONE=''
if [ -f package.json ]; then
  command -v npm >/dev/null 2>&1 || pkgi nodejs-lts
  PKG=$(node -p 'require("./package.json").name||""' 2>/dev/null)
  BINS=$(node -p 'const p=require("./package.json");typeof p.bin==="string"?p.name:Object.keys(p.bin||{}).join(" ")' 2>/dev/null)
  REPO_URL=$(npm view "${'$'}PKG" repository.url 2>/dev/null)
  if [ -n "${'$'}BINS" ] && [ -n "${'$'}PKG" ] && echo "${'$'}REPO_URL" | grep -qiF "${'$'}LF_REPO"; then
    say "npm install -g ${'$'}PKG (published on npm)"; npm install -g --no-audit --no-fund "${'$'}PKG" 2>&1 | tail -10
  else
    say "npm install"; npm install --no-audit --no-fund 2>&1 | tail -10
    npm run build --if-present 2>&1 | tail -6
    [ -n "${'$'}BINS" ] && { say "npm install -g . (adds: ${'$'}BINS)"; npm install -g --no-audit --no-fund . 2>&1 | tail -6; }
  fi
  CMDS="${'$'}CMDS ${'$'}BINS"; [ -z "${'$'}BINS" ] && CMDS="${'$'}CMDS node (in ${'$'}PWD)"; DONE=1
fi
if [ -f requirements.txt ] || [ -f pyproject.toml ] || [ -f setup.py ]; then
  command -v pip >/dev/null 2>&1 || pkgi python
  if [ -f pyproject.toml ] || [ -f setup.py ]; then say "pip install ."; pip install . 2>&1 | tail -8
  else say "pip install -r requirements.txt"; pip install -r requirements.txt 2>&1 | tail -8; fi
  CMDS="${'$'}CMDS python (in ${'$'}PWD)"; DONE=1
fi
if [ -f Cargo.toml ]; then
  command -v cargo >/dev/null 2>&1 || pkgi rust
  say "cargo install --path ."; cargo install --path . 2>&1 | tail -8
  CMDS="${'$'}CMDS (cargo binaries in ~/.cargo/bin)"; DONE=1
fi
if [ -f go.mod ]; then
  command -v go >/dev/null 2>&1 || pkgi golang
  say "go install ./..."; go install ./... 2>&1 | tail -8
  CMDS="${'$'}CMDS (go binaries in ~/go/bin)"; DONE=1
fi
if [ -z "${'$'}DONE" ] && [ -f Makefile ]; then
  pkgi make clang
  say "make"; make 2>&1 | tail -10
  DONE=1
fi
for f in install.sh setup.sh; do [ -f "${'$'}f" ] && say "This repo has ./${'$'}f. It was NOT run: read it first, then run it yourself if it's safe."; done
[ -z "${'$'}DONE" ] && say "No package file found (package.json, pyproject.toml, requirements.txt, Cargo.toml, go.mod, Makefile). Read the README for how to run it."
echo "COMMANDS:${'$'}CMDS"
""".trimStart()

    /** The whole script as Termux runs it, with the repo name available as ${'$'}LF_REPO. */
    fun fullScript(repo: String): String = "export LF_REPO=${q(repo)}\n" + installScript(repo)

    /** What the install printed after "COMMANDS:". */
    fun commandsFrom(output: String): String =
        output.lines().lastOrNull { it.startsWith("COMMANDS:") }?.removePrefix("COMMANDS:")?.trim()?.replace(Regex("\\s+"), " ") ?: ""

    fun succeeded(output: String): Boolean = output.lines().any { it.trim() == "INSTALL_OK" }

    /** The part of a README that says how to use the thing, cut to [max] characters. */
    fun usageOf(readme: String, max: Int = 1500): String {
        val text = readme.replace(Regex("<!--[\\s\\S]*?-->"), "").replace(Regex("<[^>]+>"), "")
            .lines().filterNot { it.trim().startsWith("[![") || it.trim().startsWith("![") }.joinToString("\n")
        val heading = Regex("^#{1,4}\\s*(usage|quick ?start|getting started|how to use|examples?|cli|commands)\\b.*$",
            setOf(RegexOption.IGNORE_CASE, RegexOption.MULTILINE))
        val start = heading.find(text)?.range?.first ?: 0
        return text.substring(start).replace(Regex("\n{3,}"), "\n\n").trim().let { if (it.length > max) it.take(max) + "…" else it }
    }
}
