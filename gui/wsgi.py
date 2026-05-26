#!/usr/bin/env python3
"""WSGI entrypoint for serving the WebGUI behind a production server (gunicorn).

Run from this directory, e.g.:

    gunicorn -w 1 --threads 4 -b 127.0.0.1:5001 wsgi:app

Configuration is read from the environment so the same module works for both
local and tunnel-exposed deployments:

    PLENA_EMULATOR_HOST     emulator service host  (default: 127.0.0.1)
    PLENA_EMULATOR_PORT     emulator service port  (default: 7878)
    PLENA_WEBGUI_USERNAME   basic-auth user        (default: plena)
    PLENA_WEBGUI_PASSWORD   basic-auth password    (if set, auth is enforced)

The GUI is the interactive dev console: logged-in users can load files,
execute opcode/ASM, reset, and read memory. For a public link, set
PLENA_WEBGUI_PASSWORD and confine paths with PLENA_WEBGUI_FILE_ROOT.
"""

import os

from webgui import create_app

app = create_app(
    default_emulator_host=os.environ.get("PLENA_EMULATOR_HOST", "127.0.0.1"),
    default_emulator_port=int(os.environ.get("PLENA_EMULATOR_PORT", "7878")),
)
