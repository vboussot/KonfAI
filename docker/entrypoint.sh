#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
    set -- konfai --help
fi

case "$1" in
    konfai|konfai-apps|konfai-apps-server|konfai-cluster)
        exec "$@"
        ;;
    *)
        exec konfai "$@"
        ;;
esac
