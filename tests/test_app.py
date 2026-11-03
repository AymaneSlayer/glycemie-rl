from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_runs():
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1
