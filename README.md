# kielipankki-rahti

This repository hosts code and tests for our public services on Rahti, currently just "the API", kielipankki.rahtiapp.fi.

## API

[API documentation](API.md)

### API Configuration

`openshift/` has the OpenShift configuration for everything except the route,
which is still manual.

```
openshift/
├── buildconfigs # Mostly generated; text and wav2vec need more build memory
├── deployments # Mostly generated; redis, neuralparse and text are hand-edited
├── docker # All the builds are "binary", using these dirs for context
│   ├── init_container # Downloads model data into deployments
│   ├── neuralparse
│   ├── nginx # Routes public paths to the Services defined in services/
│   ├── redis # Job store and queues
│   ├── text
│   └── wav2vec # ASR and forced alignment: one image for the API and the workers
├── imagestreams # Fully generated
├── services # Mostly generated, nginx is special
└── templates # Generates the generated confs; wav2vec_api and wav2vec_worker
              # are complete deployments parameterised by project
```

`openshift/deploy_from_scratch.sh` creates or updates everything in the
current `oc` project, builds the images and rolls the deployments out. It
takes the project from `oc project`, so the same script deploys to
`kielipankki-services-dev` and `kielipankki-services`. `SKIP="text neuralparse"`
leaves the text services out of a project that only tests audio.

Examples of using the templates and doing a build (they can either be saved as
files, like here, or fed directly to `oc`):

```
oc process -f templates/imagestreams.yaml -p IMAGE_NAME=redis | oc apply -f -
oc process -f templates/basic_buildconfig.yaml -p IMAGE_NAME=redis | oc apply -f -
oc start-build redis-build --from-dir docker/redis
oc process -f templates/basic_service.yaml -p SERVICE_NAME=redis -p SERVICE_PORT=6379 | oc apply -f -
oc process -f templates/basic_deployment.yaml -p SERVICE_NAME=redis -p SERVICE_PORT=6379 -p IMAGE_NAMESPACE=$(oc project -q) | oc apply -f -
```

### Speech recognition and alignment

`/audio/asr/fi` and `/audio/align/fi` are served by the `wav2vec` image
(`openshift/docker/wav2vec`), which runs as the `asr-api` deployment and as
the `wav2vec-fi` worker. The API converts uploads with ffmpeg, splits them on
silence and queues jobs in Redis; the worker loads a wav2vec2 CTC model from
Aalto once and serves both recognition and forced alignment from it. The
model is not in the image: `tools/package_wav2vec_model.py` zips a Hugging
Face checkpoint for the Allas bucket `kielipankki_services_data`, and an init
container fetches it. The Sámi and Finland Swedish models are fairseq
checkpoints with a custom encoder; `tools/convert_ebranchformer_to_onnx.py`
converts them once into an ONNX model directory that the same image serves
with ONNX Runtime (see the tool's docstring for the conversion environment).
`docs/wav2vec-migration.md` has the design and the remaining phases.

The image's self test (`/audio/asr/<lang>/self_test`) uses clips from
`openshift/docker/wav2vec/test/<lang>/` named `speech.mp3`, `speech.wav`,
`align.wav` and `align.txt`. They are not committed; copy them in before
building, or the self test reports its checks as `SKIPPED`. A language
without its own clips is tested with the Finnish ones.

Unit tests for the image run without a model or Redis:

```
cd openshift/docker/wav2vec && pip install -r requirements.txt fakeredis pytest onnx && pytest tests
```

TODOs:

- Move the remaining hand-edited deployments to templates or Kustomize
- Have test data in Allas
- Better automated deployment, perhaps even GitOps / ArgoCD for CD

### Tests

`test/` contains stand-alone test scripts for the live service. They all have a `--help` (which could be improved). To work, they need test data. We need to discuss where to host it.
