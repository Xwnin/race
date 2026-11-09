package com.localforge.app.agent

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RepoLogicTest {
    @Test fun parsesRepoNames() {
        assertEquals("morluto/rea", RepoLogic.parseRepo("morluto/rea"))
        assertEquals("morluto/rea", RepoLogic.parseRepo("https://github.com/morluto/rea"))
        assertEquals("morluto/rea", RepoLogic.parseRepo("(https://github.com/morluto/rea)"))
        assertEquals("morluto/rea", RepoLogic.parseRepo("github.com/morluto/rea.git"))
        assertEquals("morluto/rea", RepoLogic.parseRepo("https://github.com/morluto/rea/blob/main/README.md"))
        assertEquals("morluto/rea", RepoLogic.parseRepo("git@github.com:morluto/rea.git"))
        assertEquals("a-b/c.d_e", RepoLogic.parseRepo("a-b/c.d_e/"))
    }

    @Test fun rejectsNonRepos() {
        assertNull(RepoLogic.parseRepo("rea"))
        assertNull(RepoLogic.parseRepo(""))
        assertNull(RepoLogic.parseRepo("owner/na me"))
        assertNull(RepoLogic.parseRepo("owner/;rm -rf"))
        assertNull(RepoLogic.parseRepo("owner/.."))
    }

    @Test fun reaHasItsOwnRecipe() {
        val s = RepoLogic.fullScript("morluto/rea")
        assertTrue(s.contains("npm install -g --no-audit --no-fund rea-agents@latest"))
        assertTrue(s.contains("COMMANDS: rea"))
        assertFalse(s.contains("git clone"))
        assertTrue(s.trimEnd().endsWith("echo INSTALL_OK"))
        assertEquals(RepoLogic.REA, RepoLogic.knownFor("Morluto/REA"))
    }

    @Test fun otherReposAreClonedAndDetected() {
        val s = RepoLogic.fullScript("someone/tool")
        assertTrue(s.startsWith("export LF_REPO='someone/tool'"))
        assertTrue(s.contains("git clone --depth 1 'https://github.com/someone/tool'"))
        assertTrue(s.contains("\$HOME/lf-repos/tool"))
        assertTrue(s.contains("package.json") && s.contains("Cargo.toml") && s.contains("go.mod"))
        assertTrue(s.contains("It was NOT run"))
    }

    @Test fun readsInstallOutput() {
        val out = "== Cloning\nCOMMANDS: foo bar\n== Files\nINSTALL_OK\n"
        assertEquals("foo bar", RepoLogic.commandsFrom(out))
        assertTrue(RepoLogic.succeeded(out))
        assertFalse(RepoLogic.succeeded("CLONE_FAILED"))
    }

    @Test fun findsTheUsageSection() {
        val readme = "# Tool\n[![badge](x)](y)\nIntro text.\n\n## Install\nnpm i\n\n## Usage\nRun `tool --help`.\n"
        assertTrue(RepoLogic.usageOf(readme).startsWith("## Usage"))
        assertEquals("abc…", RepoLogic.usageOf("abcdef", 3))
    }
}
