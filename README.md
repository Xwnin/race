# LocalForge Cloud

The LocalForge Autopilot Android app (24/7 standing command, browser, tools) running big **uncensored / abliterated**
models from cloud aggregators (OpenRouter, Featherless, NanoGPT, Chutes) with your own key. Only models whose name or
description says uncensored or abliterated are listed. It can still run local GGUF models on the phone too.

Download: https://github.com/Xwnin/race/releases/download/cloud-latest/LocalForge-Cloud.apk

Built by GitHub Actions (`.github/workflows/build-apk.yml`). Keys are typed into the app and stay on the phone; never
commit them here (this repo is public).

## GitHub repos as tools (and REA)

The assistant can learn and install public GitHub repos by itself, then use them:

- `<github repo="owner/name"/>` reads the repo (description, files, README); add `file="docs/x.md"` for one file.
- `<install repo="owner/name"/>` installs it inside Termux (into `~/lf-repos/<name>`) after you tap **Install**.
  It detects npm, pip, cargo, go and make projects. It never runs a repo's own `install.sh` by itself.
- `<learn repo="owner/name">…</learn>` saves the assistant's own how-to, shown in every later chat.

[REA (Reverse Engineer Anything)](https://github.com/morluto/rea) is built in: the app knows how to install it
(`npm install -g rea-agents` with a Node version REA accepts) and how to use it on APKs, JS/Electron apps, saved
`.har` network captures, model files and other binaries. Ask e.g. *"Use REA to tell me how this APK is signed:
/sdcard/Download/app.apk"*. Termux must be set up first: Home tab → Code runners → **Set up Termux automatically**. It downloads and
installs Termux from GitHub, asks for the permission, and installs all the packages. You paste one line into Termux
once, because Android doesn't let one app change another app's settings.
