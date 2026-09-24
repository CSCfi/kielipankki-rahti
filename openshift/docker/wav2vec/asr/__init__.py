"""Speech recognition and forced alignment service on wav2vec2 CTC models.

The same package runs as the HTTP API (``asr.api``) and as a per-language
queue worker (``asr.worker``); ``asr.config`` describes the environment
variables that select the role and the model.
"""
