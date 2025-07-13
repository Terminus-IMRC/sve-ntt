#!/usr/bin/env bash

# Note: Only tested with noble.

set -euxo pipefail

release="$1"

arch="$(dpkg --print-architecture)"

dpkg --add-architecture arm64

sed -i "s/^\(Components:.*\)/&\nArchitectures: $arch/" /etc/apt/sources.list.d/ubuntu.sources

cat <<-END >>/etc/apt/sources.list.d/ubuntu.sources

	Types: deb
	URIs: http://ports.ubuntu.com/ubuntu-ports/
	Suites: $release $release-updates $release-backports $release-security
	Components: main universe restricted multiverse
	Architectures: arm64
	Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
END
