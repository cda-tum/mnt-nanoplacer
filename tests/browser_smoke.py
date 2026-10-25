"""Exercise the real GUI and worker, stopping after the first progress update."""

import json
import threading
from zipfile import ZipFile

import pytest
from playwright.sync_api import expect
from werkzeug.serving import make_server

from mnt.nanoplacer.gui import create_app


@pytest.fixture
def live_url(tmp_path):
    app = create_app(tmp_path / "runs")
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        app.extensions["nanoplacer_runs"].close()
        server.server_close()


def test_start_stop_and_export(page, live_url):
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(live_url)
    expect(page.locator("#start-button")).to_be_enabled(timeout=30_000)
    page.locator("#benchmark").select_option("trindade16")
    page.locator("#function").select_option("mux21")
    page.locator("#automatic-time-steps").uncheck()
    page.locator("#time-steps").fill("1000000")
    page.locator("#optimize").uncheck()
    page.locator("#start-button").click()
    page.wait_for_function(
        "() => Number(document.querySelector('#steps-value').textContent.replaceAll(',', '')) > 0",
        timeout=60_000,
    )
    page.locator("#cancel-button").click()
    expect(page.locator("#run-status")).to_have_attribute("data-state", "cancelled", timeout=30_000)
    with page.expect_download() as download:
        page.locator("#export-experiment").click()
    with ZipFile(download.value.path()) as archive:
        config = json.loads(archive.read("config.json"))
        status = json.loads(archive.read("status.json"))
        assert config["function"] == "mux21"
        assert status["status"] == "cancelled"
        assert status["timesteps"] > 0
        assert "rewards.csv" in archive.namelist()
    assert not errors
