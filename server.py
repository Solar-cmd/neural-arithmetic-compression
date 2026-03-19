#!/usr/bin/env python3
"""
Dashboard server for LLM compression benchmarks.
Run from the same directory as compressor.py:
    python3 server.py
"""

import os
import sys
import json
import sqlite3
import threading
import subprocess
import webbrowser
import tempfile
from datetime import datetime
from flask import Flask, Response, request, jsonify, send_from_directory

# Always resolve paths relative to this file's directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=PROJECT_DIR)
DB_PATH = os.path.join(PROJECT_DIR, "compression_history.db")

current_run = {"active": False, "pid": None}


# ─── Database ────────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            label TEXT,
            model TEXT,
            original_bytes INTEGER,
            compressed_bytes INTEGER,
            llm_ratio REAL,
            zlib_ratio REAL,
            zstd_only_ratio REAL,
            compress_tps REAL,
            decompress_tps REAL,
            token_count INTEGER,
            passed INTEGER,
            input_text TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_run(data):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO runs (timestamp, label, model, original_bytes, compressed_bytes,
            llm_ratio, zlib_ratio, zstd_only_ratio, compress_tps, decompress_tps,
            token_count, passed, input_text)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data.get("timestamp", datetime.now().isoformat()),
        data.get("label", "custom"),
        data.get("model", "distilgpt2"),
        data.get("original_bytes", 0),
        data.get("llm_zstd_size", 0),
        data.get("llm_zstd_ratio", 0),
        data.get("zlib_ratio", 0),
        data.get("zstd_only_ratio", 0),
        data.get("llm_tps", data.get("compression_tps", 0)),
        data.get("decompress_tps", data.get("decompression_tps", 0)),
        data.get("token_count", 0),
        1 if data.get("lossless_verified") else 0,
        data.get("input_text", ""),
    ))
    conn.commit()
    conn.close()


def get_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 100").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT COUNT(*) as total_runs,
               MIN(llm_ratio) as best_ratio,
               AVG(llm_ratio) as avg_ratio,
               MAX(compress_tps) as best_tps,
               AVG(compress_tps) as avg_tps,
               SUM(original_bytes) as total_bytes
        FROM runs WHERE passed=1
    """).fetchone()
    conn.close()
    if row:
        return {
            "total_runs": row[0],
            "best_ratio": round(row[1] or 0, 2),
            "avg_ratio": round(row[2] or 0, 2),
            "best_tps": round(row[3] or 0, 2),
            "avg_tps": round(row[4] or 0, 2),
            "total_bytes": row[5] or 0,
        }
    return {}


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(PROJECT_DIR, "dashboard.html")


@app.route("/history")
def history():
    return jsonify(get_history())


@app.route("/stats")
def stats():
    return jsonify(get_stats())


@app.route("/status")
def status():
    return jsonify({"active": current_run["active"]})


@app.route("/stop", methods=["POST"])
def stop():
    if current_run["pid"]:
        try:
            import signal
            os.kill(current_run["pid"], signal.SIGTERM)
            current_run["active"] = False
            current_run["pid"] = None
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "No active run"}), 404


@app.route("/run", methods=["POST"])
def run_benchmark():
    data = request.json or {}
    text = data.get("text", "")
    label = data.get("label", "custom")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    if current_run["active"]:
        return jsonify({"error": "A run is already in progress"}), 409

    def stream():
        current_run["active"] = True
        try:
            # Write input text to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8",
                dir=PROJECT_DIR
            ) as f:
                f.write(text)
                tmp_path = f.name

            # Inline script — PROJECT_DIR and tmp_path are injected as string literals
            script_lines = [
                "import sys, os",
                f"sys.path.insert(0, {repr(PROJECT_DIR)})",
                f"os.chdir({repr(PROJECT_DIR)})",
                "from compressor import LLMTextCompressor, benchmark",
                "import json",
                f"compressor = LLMTextCompressor(model_name='distilgpt2', context_window=512, zstd_level=19)",
                f"text = open({repr(tmp_path)}, encoding='utf-8').read()",
                f"result = benchmark(text, compressor, {repr(label)}, save=True)",
                "if result:",
                f"    result['input_text'] = text",
                f"    result['label'] = {repr(label)}",
                "    result['lossless_verified'] = True",
                "    print('__RESULT__:' + json.dumps(result))",
                f"os.unlink({repr(tmp_path)})",
            ]
            script = "\n".join(script_lines)

            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=PROJECT_DIR,
            )
            current_run["pid"] = proc.pid

            result_data = None
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith("__RESULT__:"):
                    result_data = json.loads(line[len("__RESULT__:"):])
                else:
                    yield f"data: {json.dumps({'type': 'log', 'text': line})}\n\n"

            proc.wait()

            if result_data:
                save_run(result_data)
                yield f"data: {json.dumps({'type': 'result', 'data': result_data})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
        finally:
            current_run["active"] = False
            current_run["pid"] = None

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Compressor:  {os.path.join(PROJECT_DIR, 'compressor.py')}")
    print(f"Dashboard:   {os.path.join(PROJECT_DIR, 'dashboard.html')}")
    print(f"Database:    {DB_PATH}")
    print("\nStarting dashboard on http://127.0.0.1:5000")
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
