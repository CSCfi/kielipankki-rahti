#!/bin/bash
# Create or update every object of the API in the current oc project, build
# the images and roll out the deployments. Run from this directory:
#
#     oc project kielipankki-services-dev
#     ./deploy_from_scratch.sh
#
# Environment:
#   LANGUAGES     space separated language codes to run workers for, default "fi"
#   MODEL_URL_<CODE>
#                 zip of that language's model, with the code upper-cased and
#                 "-" replaced by "_" (MODEL_URL_FI, MODEL_URL_SME, MODEL_URL_SV_FI);
#                 see tools/package_wav2vec_model.py. Defaults point at
#                 https://a3s.fi/kielipankki_services_data/model-<code>-wav2vec2.zip
#   WORKER_ARGS_<CODE>
#                 extra template parameters for that language's worker, for
#                 example WORKER_ARGS_SME="-p CPU_REQUEST=1 -p CPU_LIMIT=2 -p THREADS=2"
#   STAGES        space separated stage workers shared by all languages,
#                 default "diarize punctuate"; MODEL_URL_<STAGE> and
#                 WORKER_ARGS_<STAGE> work as for languages, with defaults
#                 sized for the two models (see tools/package_stage_models.py)
#   SKIP          space separated components to leave out, for example
#                 SKIP="text neuralparse" for a project that only tests audio
#
# The public Route is created by hand and is not in this repository.

set -euo pipefail
cd "$(dirname "$0")"

PROJECT=$(oc project -q)
LANGUAGES=${LANGUAGES:-fi}
STAGES=${STAGES:-diarize punctuate}
# The diarisation models are small whichever backend serves them; the
# punctuation graph is 1.1 GB in fp32, which is served because int8 changes
# some of the output.
WORKER_ARGS_DIARIZE=${WORKER_ARGS_DIARIZE:-"-p CPU_REQUEST=1 -p CPU_LIMIT=2 -p THREADS=2 -p MEMORY_REQUEST=256Mi -p MEMORY_LIMIT=1Gi -p MODEL_VOLUME_SIZE=256Mi"}
WORKER_ARGS_PUNCTUATE=${WORKER_ARGS_PUNCTUATE:-"-p CPU_REQUEST=500m -p CPU_LIMIT=2 -p THREADS=2 -p MEMORY_REQUEST=1536Mi -p MEMORY_LIMIT=3Gi -p MODEL_VOLUME_SIZE=4Gi -p QUANTIZE=none"}
SKIP=${SKIP:-}
REGISTRY=image-registry.apps.2.rahti.csc.fi

echo "Deploying to project $PROJECT"
echo "Languages: $LANGUAGES"
echo "Stages: $STAGES"

skipped() {
    [[ " $SKIP " == *" $1 "* ]]
}

# Environment variable suffix for a language code: fi -> FI, sv-FI -> SV_FI.
code_var() {
    echo "$1" | tr '[:lower:]-' '[:upper:]_'
}

model_url() {
    local var="MODEL_URL_$(code_var "$1")"
    echo "${!var:-https://a3s.fi/kielipankki_services_data/model-$(echo "$1" | tr '[:upper:]' '[:lower:]')-wav2vec2.zip}"
}

stage_model_url() {
    local var="MODEL_URL_$(code_var "$1")"
    echo "${!var:-https://a3s.fi/kielipankki_services_data/model-$1.zip}"
}

worker_args() {
    local var="WORKER_ARGS_$(code_var "$1")"
    echo "${!var:-}"
}

for lang in $LANGUAGES; do
    echo "  $lang: $(model_url "$lang") $(worker_args "$lang")"
done
for stage in $STAGES; do
    skipped "$stage" || echo "  $stage: $(stage_model_url "$stage") $(worker_args "$stage")"
done
if [ -n "$SKIP" ]; then
    echo "Skipping: $SKIP"
fi

# Deployments that exist before this run keep their old pods when oc apply
# finds nothing to change in them, and a rebuilt :latest image is only
# picked up by a restart. New deployments pull the new image on creation and
# must not be restarted in the middle of their model download.
existing=" $(oc get deployments -o jsonpath='{.items[*].metadata.name}') "

restart_existing() {
    for name in "$@"; do
        if [[ "$existing" == *" $name "* ]]; then
            oc rollout restart "deployment/$name"
        fi
    done
}

# Hand-edited manifests name the production project in their image
# references; rewrite it for the current project.
apply_manifest() {
    sed "s#$REGISTRY/kielipankki-services/#$REGISTRY/$PROJECT/#" "$1" | oc apply -f -
}

# BuildConfig names may not contain underscores, unlike imagestream names.
build_name() {
    echo "${1//_/-}-build"
}

# Wait until the newest build of each buildconfig has finished.
wait_for_builds() {
    for image in "$@"; do
        local name
        name=$(build_name "$image")
        while true; do
            phase=$(oc get builds -l "buildconfig=$name" \
                --sort-by=.metadata.creationTimestamp \
                -o jsonpath='{.items[-1:].status.phase}')
            case "$phase" in
                Complete) echo "build $image: complete"; break ;;
                Failed|Error|Cancelled) echo "build $image: $phase" >&2; exit 1 ;;
                *) sleep 10 ;;
            esac
        done
    done
}

images=()
for image in init_container neuralparse nginx redis text wav2vec; do
    skipped "$image" || images+=("$image")
done

# Imagestreams and buildconfigs

for image in "${images[@]}"; do
    oc process -f templates/imagestreams.yaml -p IMAGE_NAME="$image" | oc apply -f -
done

for image in "${images[@]}"; do
    case "$image" in
        # Hand-written: text and wav2vec need more build memory than the
        # template gives, init_container needs a name without the underscore.
        text|wav2vec|init_container) oc apply -f "buildconfigs/${image//_/-}-buildconfig.yaml" ;;
        *) oc process -f templates/basic_buildconfig.yaml -p IMAGE_NAME="$image" | oc apply -f - ;;
    esac
done

# Builds, all started at once

for image in "${images[@]}"; do
    oc start-build "$(build_name "$image")" --from-dir "docker/$image"
done

# Services

skipped neuralparse || oc process -f templates/basic_service.yaml -p SERVICE_NAME=neuralparse -p SERVICE_PORT=7689 | oc apply -f -
skipped text || oc process -f templates/basic_service.yaml -p SERVICE_NAME=text -p SERVICE_PORT=5001 | oc apply -f -
oc process -f templates/basic_service.yaml -p SERVICE_NAME=redis -p SERVICE_PORT=6379 | oc apply -f -
oc apply -f services/nginx-service.yaml
# The asr-api service is part of the wav2vec_api template.

# Deployments

wait_for_builds "${images[@]}"

apply_manifest deployments/redis-deployment.yaml
oc process -f templates/basic_deployment.yaml -p SERVICE_NAME=nginx -p SERVICE_PORT=1337 -p IMAGE_NAMESPACE="$PROJECT" | oc apply -f -
skipped neuralparse || apply_manifest deployments/neuralparse-deployment.yaml
skipped text || apply_manifest deployments/text-deployment.yaml
oc process -f templates/wav2vec_api.yaml -p IMAGE_NAMESPACE="$PROJECT" \
    -p LANGUAGES="$(echo "$LANGUAGES" | tr ' ' ',')" | oc apply -f -
for lang in $LANGUAGES; do
    # shellcheck disable=SC2046
    oc process -f templates/wav2vec_worker.yaml -p IMAGE_NAMESPACE="$PROJECT" \
        -p LANG="$lang" -p NAME="wav2vec-$(echo "$lang" | tr '[:upper:]' '[:lower:]')" \
        -p MODEL_URL="$(model_url "$lang")" $(worker_args "$lang") | oc apply -f -
done
for stage in $STAGES; do
    skipped "$stage" && continue
    # shellcheck disable=SC2046
    oc process -f templates/wav2vec_worker.yaml -p IMAGE_NAMESPACE="$PROJECT" \
        -p ROLE="$stage" -p NAME="$stage" \
        -p MODEL_URL="$(stage_model_url "$stage")" $(worker_args "$stage") | oc apply -f -
done

# Roll out the rebuilt images

for image in "${images[@]}"; do
    case "$image" in
        wav2vec|init_container) ;;
        *) restart_existing "$image" ;;
    esac
done
if ! skipped wav2vec || ! skipped init_container; then
    restart_existing asr-api
    for lang in $LANGUAGES; do
        restart_existing "wav2vec-$(echo "$lang" | tr '[:upper:]' '[:lower:]')"
    done
    for stage in $STAGES; do
        skipped "$stage" || restart_existing "$stage"
    done
fi

echo "Done. Watch the rollout with: oc get pods -w"
