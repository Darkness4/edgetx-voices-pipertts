#!/bin/sh
uv run voice-gen-piper-tts.py ./voices/en-US.csv en
uv run voice-gen-piper-tts.py ./voices/en-US_scripts.csv en/SCRIPTS
