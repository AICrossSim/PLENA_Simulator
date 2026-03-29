#!/usr/bin/env python3

import argparse
import json

from emulator_client import EmulatorClient, parse_opcode_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Python client for the transactional emulator TCP service."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Emulator service host")
    parser.add_argument("--port", type=int, default=7878, help="Emulator service port")
    parser.add_argument(
        "--opcode-file",
        help="Optional path to a machine-code file to execute via execute_file",
    )
    parser.add_argument(
        "--opcode",
        action="append",
        default=[],
        help="Optional opcode token to execute via execute_batch; may be passed multiple times",
    )
    parser.add_argument("--hbm", help="Optional HBM preload file")
    parser.add_argument("--fpsram", help="Optional FP SRAM preload file")
    parser.add_argument("--intsram", help="Optional INT SRAM preload file")
    parser.add_argument("--vram", help="Optional VRAM preload file")
    parser.add_argument(
        "--read-hbm-len",
        type=int,
        default=64,
        help="Number of HBM bytes to read back from address 0",
    )
    return parser.parse_args()

def send_request(client: EmulatorClient, request: dict[str, object]) -> dict[str, object]:
    print(f">> {json.dumps(request)}")
    response = client.request_raw(request)
    print("<<", json.dumps(response, indent=2, sort_keys=True))
    print()
    return response


def main() -> int:
    args = parse_args()
    client = EmulatorClient(host=args.host, port=args.port)

    send_request(client, {"cmd": "ping"})
    send_request(client, {"cmd": "get_config"})

    if args.hbm:
        send_request(client, {"cmd": "load_hbm_file", "path": args.hbm})
    if args.fpsram:
        send_request(client, {"cmd": "load_fp_sram_file", "path": args.fpsram})
    if args.intsram:
        send_request(client, {"cmd": "load_int_sram_file", "path": args.intsram})
    if args.vram:
        send_request(client, {"cmd": "load_vram_file", "path": args.vram})

    if args.opcode_file:
        send_request(client, {"cmd": "execute_file", "path": args.opcode_file})
    elif args.opcode:
        send_request(
            client,
            {"cmd": "execute_batch", "opcodes": parse_opcode_batch(" ".join(args.opcode))},
        )

    send_request(client, {"cmd": "get_state"})
    send_request(client, {"cmd": "read_vram", "addr": 0})
    send_request(
        client,
        {"cmd": "read_hbm", "addr": 0, "len": args.read_hbm_len},
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
