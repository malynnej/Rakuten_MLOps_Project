.PHONY: lockfiles clean_locks run_apis stop_apis clean_logs

UV := uv

# root lock target
uv.lock: pyproject.toml ${wildcard src/*/pyproject.toml}
	@echo "Generating root uv.lock..."
	$(UV) lock

# per-service lock targets
src/%/pylock.toml: uv.lock src/%/pyproject.toml
	@echo "Generating pylock.toml for $*..."
	$(UV) export --package $* --locked --format pylock.toml --output-file $@

# tests lock targets
tests/%/pylock.toml: uv.lock tests/%/pyproject.toml
	@echo "Generating pylock.toml for $*..."
	$(UV) export --package $* --locked --format pylock.toml --output-file $@

# generate lock files for each pyproject.toml in src/ and tests/
src_lockfiles := $(patsubst src/%/pyproject.toml,src/%/pylock.toml,${wildcard src/*/pyproject.toml})
test_lockfiles := $(patsubst tests/%/pyproject.toml,tests/%/pylock.toml,${wildcard tests/*/pyproject.toml})

lockfiles: uv.lock $(src_lockfiles) $(test_lockfiles)
	@echo "All lock files generated."

clean_locks:
	rm -f uv.lock src/*/pylock.toml
	rm -f uv.lock tests/*/pylock.toml

run_apis: lockfiles
	docker compose up --build

stop_apis:
	docker compose down

clean_logs:
	rm -f logs/test_predict_api.log
