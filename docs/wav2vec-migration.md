# Replacing the Kaldi ASR and aligner with wav2vec2

Status: October 2026. Phases 1 to 4 are implemented in
`openshift/docker/wav2vec`; the "as built" sections list where each
departs from its plan. Phases 2 and 3 await deployment to the test project.

## Why

`kaldi-serve` runs an old, unreproducible kaldi-serve build whose base image
exists only as a cached copy in the registry. `finnish-forced-align` drives a
Kaldi alignment script inside another third-party image and processes one
request at a time behind a directory lock. Aalto now provides wav2vec2 CTC
models for Finnish, Northern Sámi and Finland Swedish, and a CTC model gives
both recognition and forced alignment from the same forward pass. The API
already exposes word timestamps, so nothing in the public contract depends on
Kaldi.

## Phases

1. **Finnish on wav2vec2.** Replace `kaldi-serve` and `finnish-forced-align`
   with one image running as an API and a Finnish worker. Same public paths,
   same response shapes. Deploy to the test project first.
2. **Diarisation.** One shared worker, speakers added to responses.
3. **Postprocessing.** Punctuation and truecasing worker for Finnish and
   Swedish; capitalisation rules for Sámi.
4. **Sámi and Finland Swedish.** Convert the E-Branchformer checkpoints to
   ONNX, add `wav2vec-sme` and `wav2vec-sv-fi` workers from the same template,
   including alignment.

## Phase 1: Finnish on wav2vec2

### Goal and non-goals

Goal: `/audio/asr/fi/*` and `/audio/align/fi/*` behave as documented in
`API.md`, served by wav2vec2, with `kaldi-serve` and `finnish-forced-align`
gone. Existing clients, including the `tekstiks` front end and the two test
scripts, work unchanged.

Not in phase 1: speakers, punctuation, other languages, a new response format.
The code is written so that these slot in, but nothing is built for them yet.

### Components

Three deployments replace two, plus a Redis and nginx change:

| Object | Role | Notes |
|---|---|---|
| `asr-api` Deployment + Service (port 5000) | HTTP API for all `/audio/...` paths | Stateless, no model. `ROLE=api`. |
| `wav2vec-fi` Deployment | Recognition and alignment worker for Finnish | `ROLE=worker`, `ASR_LANG=fi`. Loads the model once, consumes Redis list `asr:fi`. Replicas scale throughput. |
| `redis` | Job store and queues | Now also holds audio blobs, so `maxmemory` goes up. |
| `nginx` | Router | `/audio/asr` and `/audio/align` both proxy to `asr-api:5000`. |

Both `asr-api` and `wav2vec-fi` run image `wav2vec`, built from
`openshift/docker/wav2vec/`.

### Image

```
openshift/docker/wav2vec/
  Dockerfile          python:3.12-slim, ffmpeg, CPU-only torch, transformers,
                      torchaudio, flask, gunicorn, redis, pydub
  requirements.txt    pinned
  entrypoint.sh       ROLE=api -> gunicorn asr.api:app; ROLE=worker -> python -m asr.worker
  asr/
    config.py         settings from environment (see below)
    api.py            Flask routes
    jobs.py           Redis job hashes, queues, audio blobs
    audio.py          decode any upload with ffmpeg/pydub to 16 kHz mono; silence splitting
    model.py          load the CTC model, optional int8 quantisation, emissions with chunk overlap
    ctc.py            greedy decoding, word timestamps and confidences, forced alignment
    formats.py        intervals -> ctm, eaf, TextGrid
    worker.py         BRPOP loop over the language queue, dispatch on job type
  test/               self-test clips: puhetta.mp3, puhetta.wav, pohjantuuli wav and txt
```

Environment variables, all with defaults in `config.py`:

| Variable | Meaning | Phase 1 value |
|---|---|---|
| `ROLE` | `api` or `worker` | |
| `ASR_LANG` | language code in paths and queue name | `fi` |
| `ASR_MODEL_DIR` | Hugging Face model directory | `/data/model-fi` |
| `ASR_QUANTIZE` | `none` or `int8` | `int8` |
| `ASR_THREADS` | torch intra-op threads | equal to the CPU limit |
| `ASR_CHUNK_S`, `ASR_STRIDE_S` | chunk length and overlap for long segments | 30, 5 |
| `ASR_REDIS_HOST`, `ASR_REDIS_PORT` | | `redis`, 6379 |
| `PUBLIC_BASE_URL` | used by the self test to call through nginx | `http://nginx:1337` |

Torch CPU wheels come from the PyTorch CPU index so the image stays around
1.5 GB rather than 6 GB.

### Model packaging

The worker expects a plain Hugging Face model directory (`config.json`,
`model.safetensors`, `vocab.json`, tokenizer and preprocessor configs). Zip the
directory as `model-fi-wav2vec2.zip`, upload it to the `kielipankki_services_data`
bucket next to the existing `model-fi.zip`, and have the init container fetch
and unzip it exactly as the Kaldi deployment does today. Quantisation happens
at load time in the worker; nothing quantised is stored.

Which checkpoint goes in the zip is a parameter, not a code change. Start with
`GetmanY1/wav2vec2-large-fi-475k-finetuned-experimental` (315M parameters,
8 to 10x real time on four threads with int8). The repositories are labelled
experimental; confirm with Aalto that this is the checkpoint they want served,
and get their WER comparison against xlarge on colloquial data before deciding
whether xlarge is worth three times the CPU.

### Redis changes

Job hashes keep today's fields and semantics. Additions:

- Queues: Redis lists named `asr:<lang>`. The API pushes a JSON message
  `{"jobid": ..., "task": "asr" | "align"}`; workers block on `BRPOP`.
- Audio: key `audio:<jobid>` holds the 16 kHz mono 16-bit WAV bytes the
  worker should process, with a one hour expiry. For split jobs each segment
  has its own job id and its own blob, so blobs stay small and several workers
  can process one upload in parallel. Alignment jobs hold the whole file.
- Transcripts for alignment: key `text:<jobid>`, same expiry.
- `redis.conf`: `maxmemory` from 128mb to 2gb, policy stays `volatile-ttl` so
  audio blobs are what gets evicted under pressure. The Redis deployment gets a
  memory request of 512Mi and a limit of 2560Mi. A job whose blob was evicted
  finishes with `{"error": "audio expired before processing"}` rather than
  hanging.

### Request flow

Recognition, `submit_file` (the default, split on silence):

1. API validates size and content type, decodes with pydub, converts to
   16 kHz mono 16-bit as today (ffmpeg via subprocess, same arguments).
2. API splits on silence with today's parameters and merging rule (segments of
   at least five seconds), creates the parent hash (`type=asr_segments`) and
   one child hash per segment (`type=asr`), stores each segment's WAV under
   `audio:<child>`, pushes one queue message per child, writes the `segments`
   list to the parent. This replaces the HTTP round trip through nginx that
   the current server uses to submit its own segments.
3. A worker pops a child, loads the blob, runs the model in 30 s chunks with
   5 s overlap when a segment is longer than 30 s, decodes greedily, derives
   word start and end times from frame indices and a per-word confidence from
   the mean of the maximum softmax probability over the word's frames, and
   writes `status=done`, `processing_finished` and `response` to the child
   hash, in exactly today's `responses[0]` shape.
4. `query_job` and `query_job/tekstiks` are today's code, reading the same
   hashes. The `model` field in the response changes from the Kaldi TOML spec
   to `{"name": ..., "language": "fi", "backend": "wav2vec2", "quantization":
   "int8"}`.

`submit` (raw WAV body), `segmented` and `nosplit=true` map to the same
machinery with one or many children. The synchronous `/audio/asr/fi` endpoint
enqueues and blocks polling Redis for up to 60 seconds; `responses` then holds a
single alternative. `/queue` returns `{"length": <LLEN asr:fi>}` instead of an
empty object.

Alignment, `/audio/align/fi/submit_file`:

1. API stores the converted audio and the transcript, creates the hash with
   `task=finnish-forced-align` kept for compatibility plus `type=align`, and
   enqueues `{"task": "align"}` on `asr:fi`.
2. Worker normalises the transcript to the model vocabulary: lowercase,
   punctuation removed, characters outside the vocabulary mapped by a small
   table (for example `š` to `s` for the Finnish model), digits rejected with a
   clear error in phase 1. It computes emissions over the whole file, aligns
   with `torchaudio.functional.forced_align`, merges characters into words, and
   extends each unit's end to the next unit's start.
3. Worker writes `results` as today, a JSON string with keys `ctm`, `eaf` and
   `TextGrid`, generated by `formats.py`, plus a new key `intervals`:
   `{"words": [{"start", "end", "label"}], "letters": [...]}`. The production
   files, captured in `test/reference/align_pohjantuuli.*`, have a single tier
   named `words`: the TextGrid is one `IntervalTier`, the eaf is written by
   pympi with an empty `default` tier plus `words` and no media descriptor, and
   the ctm is three columns, `start end word`, not Kaldi's five. The new
   writers reproduce these exactly and add a `letters` tier after `words` in
   the TextGrid and eaf.
4. `query_job` returns the hash as today.

Files longer than 30 minutes are refused for alignment in phase 1 with an error
message; the whole-file trellis is the limiting factor. Windowed alignment
(`ctc-segmentation`) is the follow-up if such files turn up.

Health and self test: `/audio/asr/fi/health` reports Redis and queue length,
`/audio/align/fi/health` is an alias. `/audio/asr/fi/self_test` keeps its four
checks and runs through `PUBLIC_BASE_URL` as before; a fifth check `align`
aligns the bundled pohjantuuli clip.

### nginx

In `nginx.conf`, the `/audio/asr` and `/audio/align` locations both get
`proxy_pass http://asr-api:5000;` with the existing timeouts and body size
settings. Nothing else changes. Rebuild and roll out the nginx image.

### OpenShift objects

New templates, parameterised so the same files deploy to
`kielipankki-services-dev` and `kielipankki-services`:

- `templates/wav2vec_api.yaml`: Deployment `asr-api` and Service `asr-api`
  (5000). Parameters `IMAGE_NAMESPACE`, `REPLICAS`. Resources: request 100m
  CPU and 256Mi, limit 500m and 1Gi.
- `templates/wav2vec_worker.yaml`: Deployment `wav2vec-${LANG}` with the init
  container and `emptyDir` volume. Parameters `IMAGE_NAMESPACE`, `LANG`,
  `MODEL_URL`, `REPLICAS`, `CPU_REQUEST`, `CPU_LIMIT`, `MEMORY_REQUEST`,
  `MEMORY_LIMIT`, `THREADS`, `QUANTIZE`.
- Imagestream and buildconfig for `wav2vec` from the existing templates.
- `deploy_from_scratch.sh` updated: `wav2vec` added to imagestreams, builds and
  the two new templates applied; `kaldi-serve`, `kaldi-squash` and
  `finnish-forced-align` removed.

Starting resources for `wav2vec-fi`, one replica: CPU request 2, limit 4;
memory request 2Gi, limit 5Gi; `THREADS=4`. That is within the 10 vCPU quota
once the three `kaldi-serve` pods and the aligner are gone, and one worker at
8 to 10x real time exceeds the throughput of three single-threaded Kaldi pods.
A second replica is the first knob if the queue backs up.

### Implementation order

1. Reference outputs from production are captured by
   `test/capture_reference.py` into `test/reference/` (done 2026-09-23): split,
   `nosplit` and `tekstiks` results for both clips, the raw-body `submit`
   endpoint, the synchronous endpoint, both health endpoints and one
   alignment. Re-run the script against production if the service changes
   before cutover. Two quirks it recorded that the new service should keep:
   the raw-body `submit` and the synchronous endpoint accept only a canonical
   44-byte WAV header and reject files with a `LIST` chunk, and they hand the
   WAV to the decoder unconverted, which coped with 44.1 kHz stereo. The new
   worker therefore converts whatever WAV it receives, and the header check
   may be relaxed to any parseable RIFF/WAVE file.
2. `formats.py` and `ctc.py` with unit tests against a few frames of hand-made
   emissions; TextGrid output verified to open in Praat, eaf in ELAN.
3. `model.py` and `worker.py`; run locally against a local Redis and the model
   directory, using `test/test_asr.py --local` and `test/test_align.py --local`
   through a local nginx or directly against the API port.
4. `api.py`, porting route by route from `kaldi-serve/server.py` and
   `finnish-forced-align/server.py`, keeping the response-building code.
5. Dockerfile, entrypoint, templates, model zip in Allas.
6. Deploy to `kielipankki-services-dev`: redis, nginx, asr-api, wav2vec-fi.
   Run both test scripts with `--domain` against the test route and diff the
   results against `test/reference/`, ignoring timing fields. Run the self
   test. Submit a long recording and watch memory and queue length.
7. Production: build `wav2vec`, apply the templates, roll out redis and nginx,
   scale `kaldi-serve` and `finnish-forced-align` to zero. Keep their
   cluster objects for a week, then delete them with `oc delete`. Their
   sources and manifests are already gone from the repository (the last
   commit with them is the one before the wav2vec branch), including the
   `kaldi-serve:latest-py3.6` base image reference that only exists as a
   cached copy in the production registry.
8. Update `API.md`: `model` field contents, `intervals` in alignment results,
   tier names, alignment length limit, `queue` response.

### Acceptance

- `test/test_asr.py` with and without `--nosplit`, and with
  `--query-path /tekstiks`, returns the same keys and the same segment
  structure as the references; transcripts agree up to model differences.
- `test/test_align.py` returns `results` with `ctm`, `eaf`, `TextGrid` and
  `intervals`; the `words` tier matches `test/reference/align_pohjantuuli.TextGrid`
  within 100 ms per boundary.
- `/audio/asr/fi/self_test` reports all checks `UP`.
- A one hour recording completes in under ten minutes with one worker at four
  threads, and the worker's memory stays under its limit.
- Restarting the worker mid-job leaves the job pending rather than lost: the
  queue message is re-queued on startup from a per-worker in-progress key
  (`BRPOPLPUSH` pattern), or the job is marked failed with an error. Choose
  the first unless it complicates the worker noticeably.

### Risks and open questions

- Word boundaries: greedy CTC occasionally joins or splits words
  ("pohjantuulija"). The base model did not do this on the test clip, large
  and xlarge each did once. Aalto may have decoding advice, for instance a
  small n-gram language model with `pyctcdecode`, which would slot into
  `ctc.py` later without API changes.
- Alignment of numbers and foreign characters depends on the normalisation
  table; the first real transcripts will show what else is needed.
- The current `finnish-forced-align` response is the raw Redis hash with a JSON
  string in `results`; clients therefore already double-parse. Adding
  `intervals` inside that string is the least surprising place for it.
- Confirm with Aalto which checkpoints are release quality, since all three
  Finnish repositories carry the `experimental` suffix and empty model cards.

## Phase 1 as built

Departures from the plan above, all deliberate:

- **torchaudio does the CTC algorithms.** Forced alignment is
  `torchaudio.functional.forced_align`, and `merge_tokens` turns frame paths
  (forced or greedy) into token spans; `ctc.py` only groups spans into
  words, applies the timing corrections and maps transcripts onto the
  vocabulary. Digits in alignment transcripts are spelled out with
  `num2words` for Finnish and Swedish and rejected for Sámi, which it does
  not cover. Library implementations are preferred to owned code wherever a
  library covers the operation.
- **Timing corrections.** CTC emits a character about two frames after its
  onset, so every boundary is moved 40 ms earlier (`ctc.LATENCY_S`). A
  word's last character also ends early, so a unit's end is extended to the
  next unit's start when the gap is at most 0.3 s (`ctc.MAX_GAP_S`), and by
  0.2 s before a longer pause or at the end of the audio
  (`ctc.PAUSE_EXTENSION_S`), never into the next word. Letters within a word
  are contiguous. The constants were chosen on the pohjantuuli clip against
  the Kaldi aligner: with them all 160 word boundaries of the large int8 model
  are within 100 ms of the reference (mean absolute error 30 ms); without
  them starts were 44 ms late on average and 18 of 80 ends were off by more
  than 100 ms. The same corrections apply to recognition word times.
- **Model directory.** The init container unzips wherever the zip puts
  things and moves the directory containing `config.json` to `/data/model`,
  so `ASR_MODEL_DIR` is always `/data/model` and the zip layout does not
  matter. `tools/package_wav2vec_model.py` builds the zip and adds a
  `kielipankki.json` with the source repository, which the worker reports as
  the model `name`.
- **Failures are visible.** Job hashes get `status=failed` and an `error`
  field (also as `response={"error": ...}` for recognition) when audio
  expired, the transcript cannot be normalised, or the worker hit an
  exception. `query_job` returns `status: failed` with the error;
  `tekstiks` uses its error code 1.
- **Workers announce themselves.** Each worker refreshes a heartbeat key
  while it runs, publishes its model description under `model:<lang>`, and
  keeps claimed messages in `asr:<lang>:processing:<pod>` (`BRPOPLPUSH`). On
  start it requeues its own leftovers, and whenever idle it requeues the
  lists of workers whose heartbeat has expired, so a pod deleted mid-job
  loses nothing. Health reports `workers` and `queue_length`.
- **The API knows its languages.** `ASR_LANGUAGES` (default `fi`) lists the
  codes the API accepts; adding Sámi is a template parameter, not a code
  change. `/audio/asr/health` without a language is the readiness probe.
- **Raw WAV endpoints** accept any RIFF/WAVE file and convert it, instead of
  requiring the 44-byte header.
- **nginx resolves upstreams per request.** `start.sh` in the nginx image
  writes the cluster DNS server into a `resolver` directive and the pod's
  namespace into `$svc_domain`; the `proxy_pass` targets are variables with
  fully qualified Service names, because nginx's resolver ignores DNS search
  domains. nginx therefore starts and serves the other
  paths even when a backend Service such as `text` does not exist, as in a
  test project deployed with `SKIP`.
- **Redis settings are `ASR_REDIS_HOST` and `ASR_REDIS_PORT`**, and the pods
  have `enableServiceLinks: false`: Kubernetes otherwise injects
  `REDIS_PORT=tcp://<ip>:6379` from the redis Service into every container.
- **Deployment.** `deploy_from_scratch.sh` deploys to whatever project `oc`
  is pointed at and rewrites the image namespace in the hand-edited
  manifests; the worker uses the `Recreate` strategy so a rollout does not
  need room for two model copies within the CPU quota.

Measured locally with the large model, int8, four threads: recognition at
6 to 10x real time, alignment of the 37 s clip in 0.05 s after the forward
pass, peak RSS 3.5 GB (during quantisation; the template's 5Gi limit holds).
The transcript of the clean clips matches production except for the known
"pohjantuulija" join.

Not done yet: uploading the model zip to Allas (needs Allas credentials),
the deployment itself, and the `API.md` update planned as step 8.

## Phase 2: diarisation

One `diarize` worker for all languages, running sherpa-onnx with the pyannote
segmentation 3.0 model and a speaker embedding model (about 13x real time on
four threads), or pyannote.audio 4 community-1 if its accuracy is needed and
the gated download and heavier dependencies are acceptable. The API
enqueues `diarize` alongside the recognition segments when `diarize` is not
`false`; `query_job` assigns each word to the turn with maximal overlap;
`tekstiks` fills `speakers` and turns instead of the fixed `S0`. Diarisation
failure degrades to a single speaker and a `checks` entry.

## Phase 3: postprocessing

One `punctuate` worker running the `1-800-BAD-CODE/xlm-roberta_punctuation_
fullstop_truecase` ONNX model, as a separate queue stage after recognition
and diarisation, when `punctuate` is not `false`. Responses gain
`transcript_punctuated` next to the raw transcript; the raw transcript stays
as is. Only the ONNX and SentencePiece files are packaged, the repository
also carries an unneeded 1.1 GB `.nemo` file. Languages the model does not
cover get sentence-initial capitalisation from pause and turn boundaries.

### Phases 2 and 3 as built (2026-10-01)

- **Stage jobs.** A recognition job submitted with `diarize` or `punctuate`
  (both default to on; `false` declines) gets one job hash per stage, named
  in its `diarize_job` and `punctuate_job` fields. The diarisation job has
  its own copy of the whole audio and is queued on `asr:diarize` at submit
  time, so it runs in parallel with recognition. The punctuation job is
  created at submit time but queued on `asr:punctuate` only when every
  recognition segment and the diarisation have settled: `stages.settle`
  runs after every job in every worker and after splitting, and a
  set-if-absent field on the punctuation job makes sure it is queued once.
  The job as a whole stays `pending` until its stages have settled, so
  clients that stop polling at `done` see speakers and punctuation.
- **Stages without a worker are skipped, not waited for.** At submit time
  the API checks the stage's worker heartbeats; with none alive the stage
  job is written as `skipped` at once. A stage that fails leaves the job
  `done` without its output. Either case is reported in `checks`
  (`UP`, `DOWN`, `SKIPPED`, or `PENDING` in a partial result) and
  `warnings` (the reason) of the query responses.
- **One image, four roles.** `ROLE=diarize` and `ROLE=punctuate` run the
  same `asr.worker` loop with a different handler; the worker template
  takes `ROLE`, and `deploy_from_scratch.sh` deploys the stages listed in
  `STAGES` with `WORKER_ARGS_DIARIZE` and `WORKER_ARGS_PUNCTUATE` sized for
  the models (2 CPU and 1Gi; 2 CPU and 3Gi). `tools/package_stage_models.py`
  builds `model-diarize.zip` and `model-punctuate.zip` for the init
  container, which finds the model directory by its `kielipankki.json`.
- **Embedding model: TitaNet, not WeSpeaker.** The first deployment used
  the WeSpeaker ResNet34 embeddings and labelled the whole Finland Swedish
  test interview as one speaker, with the second voice appearing only as
  one-second interjections; the same happened at any clustering threshold.
  Of the embedding models sherpa-onnx redistributes, NVIDIA's TitaNet
  small found the interviewer's turns (0 to 5 s, 21 to 29 s, 48 to 50 s)
  with the speaker count fixed at two and three speakers unconstrained;
  3D-Speaker's CAM++ came close, ERes2Net was slower for no gain. On the
  Sámi interview all of them agree once the count is fixed, so that clip
  cannot rank them. `tools/package_stage_models.py` packages TitaNet by
  default and takes `--embedding` for the others. Diarisation of the 75 s
  clip takes 5 s on four threads with TitaNet against 7 s with WeSpeaker.
- **pyannote.audio backend (2026-10-05).** `load_diarizer` picks
  `PyannoteDiarizer` when the model directory holds a pyannote pipeline
  (`config.yaml`, or `backend: pyannote` in `kielipankki.json`), otherwise
  `SherpaDiarizer`; both return the same turn list. The pyannote backend
  is meant for `pyannote/speaker-diarization-community-1` (September
  2025, CC BY 4.0, gated: accept the terms on huggingface.co and run
  `tools/package_stage_models.py diarize --backend pyannote` with
  `HF_TOKEN` set). It reports better speaker counting than the 3.1
  pipeline that sherpa-onnx re-implements, and its exclusive diarisation
  has no overlapping turns, so word assignment never has to arbitrate
  between nested turns; the query-time smoothing of speaker labels that
  the nested turns had needed was removed again. pyannote.audio 4 sends
  usage telemetry unless `PYANNOTE_METRICS_ENABLED=false`, which the
  Dockerfile sets. The pipeline's default batch size of 32 windows per
  forward pass was killed by a 2Gi memory limit on the first 75 s clip;
  the worker sets both batch sizes to `ASR_DIARIZE_BATCH` (4), since
  batching gains nothing on CPU. Not yet measured on real audio: the models
  are gated, so the first run is in the test project.
- **Diarisation output.** The worker writes `{"turns": [{"speaker",
  "start", "end"}], "speakers"}` with labels `S0`, `S1`, ... in order of
  first appearance. sherpa-onnx's turns can nest (a short interjection inside
  a long turn), so a word takes the turn it overlaps most and, on a tie, the
  shorter one; a word overlapping no turn takes the nearest. An optional
  `num_speakers` request parameter fixes the number of clusters; otherwise
  the clustering threshold (`ASR_DIARIZE_THRESHOLD`, 0.5) decides. Measured
  locally with four threads: 10x real time, 200 MB of models.
- **Punctuation output.** Every word gains `word_with_punctuation` and
  `punctuation`, the names of the Estonian pipeline's format, and every
  alternative gains `transcript_punctuated`. The punctuation worker
  punctuates the whole job's words as one text, so sentences run across
  segment boundaries and the model has all the context, then capitalises
  the first word of each turn (speaker change or two seconds of silence),
  as the Estonian pipeline does. Punctuating turn by turn was tried first
  and chopped the stretches where one speaker's short interjections nest in
  the other's turn into one-word sentences. The `punctuators` package runs
  the model; its output is mapped back word by word, and a word whose
  letters the output does not reproduce keeps its own letters. ONNX Runtime's thread count is set by patching the session
  constructor the package calls, which offers no other way.
- **Swedish uses the model after all.** The model's 47 languages include
  Finnish but not Swedish; tried zero-shot on the Finland Swedish test
  transcript it produced sentence breaks, questions and the capitals of
  Helsingfors and OS, so `ASR_PUNCTUATE_MODEL_LANGUAGES` defaults to
  `fi,sv` (matched on the first subtag, so `sv-FI` counts). On Sámi text it
  capitalised random words, so Sámi and any other language get the rules:
  a capital at the start of a turn and after a pause of at least
  `ASR_PUNCTUATE_PAUSE_S` (1 s), and a full stop before the pause and at the
  end of the turn.
- **fp32, both graphs shipped.** The zip holds the fp32 graph (1.1 GB) and
  an int8 copy (280 MB, MatMul and Gather weights). Int8 changed 4 of 80
  Finnish words and 25 of 224 Swedish words, some for the worse (`OS` to
  `os`, lost commas), and the model takes well under a second per minute of
  speech either way, so the punctuation worker runs `QUANTIZE=none`: 2.1 GB
  peak RSS against 0.8 GB for int8.
- **tekstiks turns.** Sections keep `start`, `end` and the raw
  `transcript` and gain `type` and `turns`, each `{"speaker", "start",
  "end", "transcript", "words"}` as in the Estonian pipeline's
  `resegment_json.py`: absolute times, the transcript built from
  `word_with_punctuation`, a turn ending where the next begins, and
  `speakers` listing every label. Turns do not cross sections. At the
  tekstiks developer's request (2026-10-06) the section-level `words`,
  which duplicated the turns' words with segment-relative times, were
  dropped, and a section with no recognised words is `type: "non-speech"`
  with no turns.
- **A job is attempted at most twice** (`ASR_MAX_ATTEMPTS`). Workers
  requeue the jobs they were holding when they start, so a job that kills
  its worker, as the first pyannote job did when the pod's memory limit was
  still 1Gi, would otherwise be retried for ever; the second death fails it
  with `processing failed repeatedly`.
- **Health** gains `stages` with the worker count and queue length of
  `diarize` and `punctuate`; the self test reports `diarization` and
  `punctuation` as the query result does.

## Phase 4: Sámi and Finland Swedish

Decision, 2026-09-24: serve the checkpoints Aalto recommended,
`GetmanY1/wav2vec2-large-ebranch-sami-18k-finetuned-experimental` and
`GetmanY1/wav2vec2-large-ebranch-finswe-86k-finetuned-experimental`. The
service is meant to show the models at their best, so the stock
`wav2vec2-large-sami-cont-pt-22k-finetuned`, which the current image could
serve without new code, stays a documented fallback rather than a shortcut,
and no WER comparison is requested first. For Finland Swedish there is no
alternative in any case: `wav2vec2-large-FS` is a pretrained encoder without a
CTC head.

Both checkpoints are fairseq models with an E-Branchformer encoder and load
only with fairseq from git, hydra-core 1.0.7, a patched omegaconf
(`Getmany1/omegaconf@2.0_branch`) and Python 3.10, plus the
`fairseq_extra_encoders` package shipped in each model repository. Aalto's
demo Space (`GetmanY1/sami_asr`) shows the reference usage and settles a few
details for whichever backend serves them:

- The vocabulary is the four specials `<s> <pad> </s> <unk>` at indices 0 to
  3 followed by `dict.ltr.txt` in order, `|` is the word boundary, and the
  CTC blank is index 0, not `<pad>` as in the Hugging Face models.
- Whether audio is normalised to zero mean and unit variance is the
  checkpoint's `cfg.task.normalize`; it has to travel with the exported model.
- Their long-audio inference uses the same 30 s windows with 5 s stride as
  our worker.

Two ways to serve them, with the plan being the first:

1. **Convert to ONNX once** with `tools/convert_ebranchformer_to_onnx.py`,
   run in a conversion environment (Python 3.10, torch, hydra-core 1.0.7,
   the patched omegaconf, fairseq from git, onnx, onnxruntime): it loads the
   checkpoint with fairseq, replaces the sinusoidal relative position table
   (100 000 positions, 614 MB) with one sized for the longest window, exports
   the encoder and CTC projection at opset 17 with a dynamic time axis,
   quantises the weights to int8 with ONNX Runtime, checks the fp32 and int8
   graphs against torch on a test clip, and zips `model.onnx`,
   `dict.ltr.txt` and a `kielipankki.json` carrying the name, the
   `normalize` flag and the frame ratio. `model.py` has an `OnnxRecognizer`
   next to the Hugging Face one behind the same `emissions` interface;
   `load_recognizer` picks it when the model directory holds `model.onnx`.
   Only `MatMul` nodes are quantised: quantising the feature extractor's
   convolutions produces `ConvInteger` nodes that ONNX Runtime's CPU
   provider cannot run. Measured on the Sámi model, four threads: the fp32
   export matches torch frame for frame, int8 agrees with fp32 on 98.7 % of
   frames and runs at 3.7x real time against 1.9x for fp32 (1.4x for torch);
   the int8 file is 330 MB against 1.2 GB. fairseq is a conversion-time
   dependency only.
2. **Serve with fairseq directly**, as the demo does: a second image built
   like the demo's Dockerfile with a `fairseq` backend. No conversion step,
   but a fragile pinned stack and the slower model. The fallback if the ONNX
   export of the E-Branchformer turns out not to be faithful.

Deployment is otherwise configuration: the worker template takes `LANG`,
`MODEL_URL` and a `NAME` (Deployment names must be lowercase, so `sv-FI`
becomes `wav2vec-sv-fi`), `deploy_from_scratch.sh` loops over `LANGUAGES`, and
the API's `ASR_LANGUAGES` lists the codes. nginx needs no change: the
`/audio/asr` and `/audio/align` locations already cover every language.
Self-test clips live in `openshift/docker/wav2vec/test/<lang>/` as
`speech.mp3`, `speech.wav`, `align.wav` and `align.txt`; a language without
its own directory is tested with the Finnish clips and the response says so
in `test_clips`. Sámi and Swedish clips are still to be added, and Sámi gets
the rule-based capitalisation of phase 3.

### Phase 4 as built (2026-09-24)

- `tools/convert_ebranchformer_to_onnx.py` converted both checkpoints in a
  Python 3.10 environment built as described in its docstring. The plugin
  modules shipped with the two repositories are identical. Each zip
  (`model-sme-wav2vec2.zip`, `model-sv-fi-wav2vec2.zip`, about 1.4 GB)
  holds `model.onnx` (int8, 334 MB), `model.fp32.onnx` (1.2 GB),
  `dict.ltr.txt` and `kielipankki.json`; the worker's `QUANTIZE` parameter
  (`int8` or `none`) picks the graph, so precision is a deployment choice.
- Measured on the Finnish test clip, which is out of domain for both models
  and serves only as a mechanical check: the ONNX backend loads in one
  second, runs int8 at 5.7x real time on four threads with a 1.3 GB peak,
  and alignment through the same emissions works, including the digit and
  character checks against the fairseq dictionary. The int8 graphs agree
  with fp32 on 98.7 % (Sámi) and 95.3 % (Swedish) of frames on that clip.
- Nothing has been run on Sámi or Swedish speech yet. The first real
  recordings decide between int8 and fp32 and whether the timing constants
  chosen on Finnish (`ctc.LATENCY_S` and friends) hold for the
  E-Branchformer models.


Quota: three workers at the Finnish worker's 4-core limit are 12 cores of
`limits.cpu` against the 10 available, before the API, Redis, nginx and the
text services. Sámi and Swedish traffic is small, so 2 cores and 2 threads
each fits at roughly 8 cores for the workers, with no headroom left for
`text` and `neuralparse`; `deploy_from_scratch.sh` takes per-language
resource overrides in `WORKER_ARGS_<CODE>`. Request more quota before phase
4 goes to production.
