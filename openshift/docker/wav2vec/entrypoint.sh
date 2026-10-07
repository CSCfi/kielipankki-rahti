#!/bin/sh
# ROLE=api serves HTTP on port 5000; ROLE=worker consumes the queue of
# ASR_LANG; ROLE=diarize and ROLE=punctuate consume the stage queues of those
# names. Settings are read from the environment by asr/config.py.
set -e

export OMP_NUM_THREADS="${ASR_THREADS:-4}"

case "${ROLE:-api}" in
    api)
        exec gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 8 \
            --timeout 3600 --access-logfile - asr.api:app
        ;;
    worker|diarize|punctuate)
        exec python -m asr.worker
        ;;
    *)
        echo "unknown ROLE '${ROLE}', expected api, worker, diarize or punctuate" >&2
        exit 1
        ;;
esac
