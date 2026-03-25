test:
	./.venv/bin/python -m unittest discover -s tests
verbose-test:
	./.venv/bin/python -m unittest discover -s tests -v
release-smoke:
	./.venv/bin/python scripts/release_smoke.py --uv-cache-dir /tmp/uv-cache
