#!/bin/sh
# Builds dist/splat-<version>-macos-arm64.tar.gz: a standalone Python with splat and its locked dependencies.
set -eu

cd "$(dirname "$0")/.."
version="$(uv version --short)"
name="splat-$version-macos-arm64"
work="build/bundle"
root="$work/$name"

rm -rf "$work" && mkdir -p "$root/bin" dist

uv build --wheel --out-dir "$work/wheel"
uv python install 3.12 --install-dir "$work/python" --no-bin
# the install dir also holds a minor-version symlink, so take the real directory
mv "$(find "$work/python" -maxdepth 1 -type d -name 'cpython-3.12.*')" "$root/python"
python="$root/python/bin/python3.12"

uv export --frozen --no-dev --no-emit-project --format requirements-txt --quiet > "$work/requirements.txt"
install() { uv pip install --python "$python" --break-system-packages --link-mode copy --compile-bytecode "$@"; }
install -r "$work/requirements.txt"
install --no-deps "$work"/wheel/*.whl

# console scripts hardcode the build path; bin/splat replaces them
find "$root/python/bin" ! -type d ! -name 'python3*' -delete

cat > "$root/bin/splat" <<'EOF'
#!/bin/sh
here="$(dirname "$(readlink -f "$0")")"
exec "$here/../python/bin/python3.12" -I -c 'import sys; from splat.cli.main import main; sys.argv[0] = "splat"; sys.exit(main())' "$@"
EOF
chmod +x "$root/bin/splat"

tar -C "$work" -czf "dist/$name.tar.gz" "$name"
(cd dist && shasum -a 256 "$name.tar.gz" > "$name.tar.gz.sha256")

# the unpacked copy must run from another path with an empty environment
check="$work/check"
mkdir -p "$check" && tar -C "$check" -xzf "dist/$name.tar.gz"
env -i HOME="$HOME" PATH=/usr/bin:/bin "$check/$name/bin/splat" --version | grep -qx "$version"
env -i HOME="$HOME" PATH=/usr/bin:/bin "$check/$name/bin/splat" doctor
echo "dist/$name.tar.gz"
