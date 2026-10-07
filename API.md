# kielipankki.rahtiapp.fi API reference

## Text endpoints

The following endpoints accept running text as the data payload, and directly return a json response, which represents a list of sentences, with each sentence comprising a list of tokens. The tokens are the direct output of the underlying tools, but a more structured format is planned. Eg:

	$ curl --data "Voi voi olla." kielipankki.rahtiapp.fi/text/fi/postag | jq
	[
	  [
	    [
	      "Voi",
	      "voi",
	      "[POS=PARTICLE]|[SUBCAT=INTERJECTION]"
	    ],
	    [
	      "voi",
	      "voida",
	      "[POS=VERB]|[VOICE=ACT]|[MOOD=INDV]|[TENSE=PRESENT]|[PERS=SG3]"
	    ],
	    [
	      "olla",
	      "olla",
	      "[POS=VERB]|[VOICE=ACT]|[INF=A]|[CASE=LAT]"
	    ],
	    [
	      ".",
	      ".",
	      "[POS=PUNCTUATION]"
	    ]
	  ]
	]


#### `/text/fi/postag` (POST)

#### `/text/fi/nertag` (POST)

#### `/text/fi/sentiment` (POST)

#### `/text/fi/annotate` (POST)

The following endpoints, intended to support slightly larger jobs that may time out, support a submit-and-query scheme. Eg:

	$ curl --data '@text_to_process.txt' kielipankki.rahtiapp.fi/text/fi/nertag/submit
	{"jobid":"4afa7d86-3416-4993-8110-ab9a7e2de39e"}
	$ curl --data '4afa7d86-3416-4993-8110-ab9a7e2de39e' kielipankki.rahtiapp.fi/text/fi/nertag/query_job
	...verbose output...

#### `/text/fi/sentiment/submit` (POST)

#### `/text/fi/sentiment/query_job` (POST)

## Audio endpoints

### ASR

Endpoints for automatic speech recognition.

The paths below are for Finnish; `/audio/asr/sme/...` and `/audio/asr/sv-FI/...` serve Northern Sámi and Finland Swedish in the same way.

#### `/audio/asr/fi/submit_file` (POST)

Submit a form with a `file` key, eg. `curl -F 'file=@audio.mp3' http://kielipankki.rahtiapp.fi/audio/asr/fi/submit_file`. The response is a json object containing a `jobid` key, which is used later for polling for results with `/audio/asr/fi/query_job`.

Optional parameters, as query string or form fields, accepted by every submission endpoint:

* `nosplit=true` recognises the file as one segment instead of splitting it on silence.
* `diarize=false` turns off speaker diarisation, `punctuate=false` turns off punctuation and truecasing. Both are on by default.
* `num_speakers=N` fixes the number of speakers for diarisation; by default it is detected.
  
Example output:
  
	{"file":"puhetta.mp3","jobid":"357f3518-afaa-45e9-bda7-b52a60b73000"}
  
Additional fields may appear, the `jobid` field is the important one.
  
#### `/audio/asr/fi/submit` (POST)

Submit a wav file as the data payload, eg. `curl --data-binary @audio.wav https://kielipankki.rahtiapp.fi/audio/asr/fi/submit`. The response is a json object containing a `jobid` key, which is used later for polling for results with `/audio/asr/fi/query_job`.
  
Example output:
  
	{"jobid":"337adadb-37ff-4492-9480-d2ffb1126932"}
  
#### `/audio/asr/fi/query_job` (POST)

Submit a jobid as the data payload. The response may be a partial result, a complete result, a pending result, or an error state.
  
##### Error states

If the submitted job id is unknown to the service, meaning that no such job has been submitted, or has been submitted so long ago that the cache has forgoten about it, the service returns

`{"error": "job id not available"}`

This is also the response if the job has multiple segments, and one or more of those segments has an unknown job id.

##### Pending result

In this case the job is known, but there is no result to report yet, partial or otherwise.

`{'status': 'pending'}`

##### Complete result

A `status` of `done` with the recognised `segments`, each holding `responses` with a `transcript`, a `confidence` and timed `words`, plus the `model` that was used. Word times are relative to the segment, segment `start` and `stop` to the file. A segment's `type` is `speech` when words were recognised in it and `non-speech` otherwise; a job submitted without splitting carries the same key at the top level.

When diarisation ran, every word has a `speaker` label (`S0`, `S1`, ...) and the result has `diarization` with the `speakers` and the speaker `turns` as `start`, `end` and `speaker`. Words take the turn they overlap most; the segments themselves come from the silence splitting and can contain several speakers. When punctuation ran, every word has `word_with_punctuation` (cased, with any attached punctuation) and `punctuation` (the marks following the word, possibly empty), and every alternative has `transcript_punctuated` next to the raw `transcript`.

`checks` reports each requested stage as `UP`, `DOWN` (it failed; the job is still done, without that output) or `SKIPPED` (no worker for it was running), and `warnings` gives the reason for anything not `UP`. The job stays `pending` until its stages have finished.

#### `/audio/asr/fi/query_job/tekstiks`

This is a specialised endpoint that conforms to a particular front-end.

Submit a jobid as the data payload. The response may be a partial result, a complete result, a pending result, or an error state.

The `result` holds `speakers` (one key per label) and `sections`, one per segment of the audio, with `type`, `start` and `end`. A section in which words were recognised has `type: "speech"`, the raw `transcript` and `turns`; one without is `type: "non-speech"` and has nothing else. A turn is one speaker's stretch of speech within the section: `speaker`, `start`, `end`, `transcript` (punctuated when punctuation ran) and `words`, with times relative to the file as in the format of the Estonian ASR pipeline.

### Forced alignment

Endpoints for aligning a known transcript with an audio to obtain timestamps.

#### `/audio/align/fi/submit_file`

Submit a form with `audio` and `transcript` keys.

#### `/audio/align/fi/query_job`

## Health checks and tests

The following endpoints under `kielipankki.rahtiapp.fi` may be used as health checks for their respective services:

* `/health` (the web server)
* `/audio/asr/fi/health`
* `/audio/align/fi/health`
* `/text/fi/health`
* `/text/fi/parse/health`

These should all return a `200` code when healthy in less than one second. With one exception, they will also return a JSON response with the field `status` having the value `UP` when healthy, and `DOWN` when unhealthy. They may optionally also contain a `checks` field, the value of which is another JSON object with name-status keypairs. The exception is `/text/fi/parse/health`, which will only return a `200` response with a nonmeaningful text response.

Example:

`{"checks":{"redis":"UP","workers":"UP"},"queue_length":0,"workers":1,"stages":{"diarize":{"workers":1,"queue_length":0},"punctuate":{"workers":1,"queue_length":0}},"status":"UP"}`

The following endpoints perform more thorough tests of their functioning. These tests should not be called too frequently (eg. once a day is ok, or more often if the service has recently failed).

* `/audio/asr/fi/self_test`
* `/text/fi/self_test`

Example:

```bash
$ curl kielipankki.rahtiapp.fi/audio/asr/fi/self_test
{"checks":{"align":"UP","decoding":"UP","diarization":"UP","punctuation":"UP","query_response":"UP","redis":"UP","submit_file":"UP","workers":"UP"},"status":"UP"}
```
