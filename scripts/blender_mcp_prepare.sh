#!/usr/bin/env bash
# Downloads the latest official Blender MCP release (Blender Lab's Gitea, not a
# git clone) and unpacks it into .blender-mcp/ — gitignored, machine-local.
set -euo pipefail

repo_api="https://projects.blender.org/api/v1/repos/lab/blender_mcp/releases"
dest="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.blender-mcp"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Fetching latest blender_mcp release metadata..."
release=$(curl -sL -A "Mozilla/5.0" "$repo_api" | jq '[.[] | select(.prerelease == false)][0]')
tag=$(echo "$release" | jq -r '.tag_name')
addon_url=$(echo "$release" | jq -r '.assets[] | select(.name | test("^mcp-.*\\.zip$")) | .browser_download_url')
server_url=$(echo "$release" | jq -r '.assets[] | select(.name | test("^blender-.*\\.mcpb$")) | .browser_download_url')

echo "Latest release: $tag"
echo "  addon:  $addon_url"
echo "  server: $server_url"

curl -sL -A "Mozilla/5.0" -o "$tmp/addon.zip" "$addon_url"
curl -sL -A "Mozilla/5.0" -o "$tmp/server.mcpb" "$server_url"

rm -rf "$dest"
mkdir -p "$dest/server"
cp "$tmp/addon.zip" "$dest/addon-${tag#v}.zip"
unzip -oq "$tmp/server.mcpb" -d "$dest/server"
echo "$tag" > "$dest/VERSION"

cat <<EOF

Installed to $dest (version $tag).

Next steps:
  1. In Blender: Edit > Preferences > Get Extensions > (dropdown) > Install from Disk...
     Select: $dest/addon-${tag#v}.zip
     Enable the "MCP" add-on, then confirm "Allow Online Access" if prompted.
  2. Register the MCP server with Claude Code (run once):
     claude mcp add blender-mcp --scope local -- uv run --project "$dest/server" blender-mcp
EOF
