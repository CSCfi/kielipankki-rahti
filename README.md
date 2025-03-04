# kielipankki-rahti

This repository hosts code and tests for our public services on Rahti, currently just "the API", kielipankki.rahtiapp.fi.

## API

[API documentation](API.md)

### API Configuration

`openshift/` has the OpenShift configuration for everything except the route, which is still manual. Also, the Dockerfile in `kaldi-serve` refers to a cached copy of `kaldi-serve:latest-py3.6`, which I put there manually, but that is also on Docker.

(Yes, these things could be fixed.)

```
openshift/
├── buildconfigs # Mostly generated, text needs an increased mem limit
├── deployments # Mostly generated, kaldi and text are special, neuralparse has an increased mem limit
├── docker # All the builds are "binary", using these dirs for context
│   ├── finnish-forced-align
│   ├── init_container # For help populating models and other data into deployments
│   ├── kaldi-serve # Relies on Allas and init_container for data
│   ├── kaldi-squash
│   ├── neuralparse
│   ├── nginx # sends traffic to the "hosts" visible only within the project defined in services/
│   ├── redis
│   └── text
├── imagestreams # Fully generated
├── services # Mostly generated, nginx is special
└── templates # Generates the generated confs

```

`openshift/deploy_from_scratch.sh` contains an example of how everything is set up. It should not be executed blindly at this point.

Examples of using the templates and doing a build (they can either be saved as files, like here, or fed directly to `oc`:

```
oc process -f templates/imagestreams.yaml -p IMAGE_NAME=redis | oc apply -f -
oc process -f templates/basic_buildconfig.yaml -p IMAGE_NAME=redis | oc apply -f -
oc start-build redis-build --from-dir docker/redis
oc process -f templates/basic_service.yaml -p SERVICE_NAME=redis -p SERVICE_PORT=6379 | oc apply -f -
oc process -f templates/basic_deployment.yaml -p SERVICE_NAME=redis -p SERVICE_PORT=6379 | oc apply -f -
```

TODOs:

- Move everything to templates or Kustomize to support deploying dev to `kielipankki-services-dev` and prod to `kielipankki-services`
- Have test data in Allas
- Better automated deployment, perhaps even GitOps / ArgoCD for CD

### Tests

`test/` contains stand-alone test scripts for the live service. They all have a `--help` (which could be improved). To work, they need test data. We need to discuss where to host it.
