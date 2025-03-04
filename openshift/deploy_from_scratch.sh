#!/bin/bash

# imagestreams

declare -a auto_imagestreams=("finnish-forced-align"
			      "init_container"
			      "kaldi-serve"
			      "neuralparse"
			      "nginx"
			      "redis"
			      "text")

for imagestream in "${auto_imagestreams[@]}"; do
    oc process -f templates/imagestreams.yaml -p IMAGE_NAME=$imagestream | oc apply -f -
done

# buildconfigs

declare -a auto_buildconfigs=("finnish-forced-align"
			      "init_container"
			      "kaldi-serve"
			      "neuralparse"
			      "nginx"
			      "redis")

for buildconfig in "${auto_buildconfigs[@]}"; do
    oc process -f templates/basic_buildconfig.yaml -p IMAGE_NAME=$buildconfig | oc apply -f -
done

oc process -f buildconfigs/text-buildconfig.yaml | oc apply -f -

# build

declare -a builds=("finnish-forced-align"
		   "init_container"
		   "kaldi-serve"
		   "neuralparse"
		   "nginx"
		   "redis"
		   "text")

for build in "${builds[@]}"; do
    oc start-build $build-build --from-dir docker/$build
done

# services

declare -a auto_services=("-p SERVICE_NAME=finnish-forced-align -p SERVICE_PORT=5003"
			  "-p SERVICE_NAME=kaldi-serve -p SERVICE_PORT=5002"
			  "-p SERVICE_NAME=neuralparse -p SERVICE_PORT=7689"
			  "-p SERVICE_NAME=redis -p SERVICE_PORT=6379"
			  "-p SERVICE_NAME=text -p SERVICE_PORT=5001")
for service in "${auto_services[@]}"; do
    oc process -f templates/basic_service.yaml "$service" | oc apply -f -
done

oc process -f services/nginx-service.yaml | oc apply -f -

# deployments

declare -a auto_deployments=("-p SERVICE_NAME=finnish-forced-align -p SERVICE_PORT=5003"
			     "-p SERVICE_NAME=nginx -p SERVICE_PORT=1337"
			     "-p SERVICE_NAME=redis -p SERVICE_PORT=6379")
for deployment in "${auto_deployments[@]}"; do
    oc process -f templates/basic_deployment.yaml "$deployment" | oc apply -f -
done

oc process -f deployments/kaldi-buildconfig.yaml | oc apply -f -
oc process -f deployments/neuralparse-buildconfig.yaml | oc apply -f -
oc process -f deployments/text-buildconfig.yaml | oc apply -f -
