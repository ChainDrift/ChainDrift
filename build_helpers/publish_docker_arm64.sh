#!/bin/sh

# Use BuildKit, otherwise building on ARM fails
export DOCKER_BUILDKIT=1

IMAGE_NAME=chaindriftorg/chaindrift
CACHE_IMAGE=chaindriftorg/chaindrift_cache
GHCR_IMAGE_NAME=ghcr.io/chaindrift/chaindrift

# Replace / with _ to create a valid tag
TAG=$(echo "${BRANCH_NAME}" | sed -e "s/\//_/g")
TAG_PLOT=${TAG}_plot
TAG_CHAINAI=${TAG}_chainai
TAG_CHAINAI_RL=${TAG_CHAINAI}rl
TAG_CHAINAI_TORCH=${TAG_CHAINAI}torch
TAG_PI="${TAG}_pi"

TAG_ARM=${TAG}_arm
TAG_PLOT_ARM=${TAG_PLOT}_arm
TAG_CHAINAI_ARM=${TAG_CHAINAI}_arm
TAG_CHAINAI_RL_ARM=${TAG_CHAINAI_RL}_arm

echo "Running for ${TAG}"

# Add commit and commit_message to docker container
echo "${GITHUB_SHA}" > chaindrift_commit

if [ "${GITHUB_EVENT_NAME}" = "schedule" ]; then
    echo "event ${GITHUB_EVENT_NAME}: full rebuild - skipping cache"
    # Build regular image
    docker build -t chaindrift:${TAG_ARM} .

else
    echo "event ${GITHUB_EVENT_NAME}: building with cache"
    # Build regular image
    docker pull ${IMAGE_NAME}:${TAG_ARM}
    docker build --cache-from ${IMAGE_NAME}:${TAG_ARM} -t chaindrift:${TAG_ARM} .

fi

if [ $? -ne 0 ]; then
    echo "failed building multiarch images"
    return 1
fi

docker build --build-arg sourceimage=chaindrift --build-arg sourcetag=${TAG_ARM} -t chaindrift:${TAG_PLOT_ARM} -f docker/Dockerfile.plot .
docker build --build-arg sourceimage=chaindrift --build-arg sourcetag=${TAG_ARM} -t chaindrift:${TAG_CHAINAI_ARM} -f docker/Dockerfile.chainai .
docker build --build-arg sourceimage=chaindrift --build-arg sourcetag=${TAG_CHAINAI_ARM} -t chaindrift:${TAG_CHAINAI_RL_ARM} -f docker/Dockerfile.chainai_rl .

# Tag image for upload and next build step
docker tag chaindrift:$TAG_ARM ${CACHE_IMAGE}:$TAG_ARM
docker tag chaindrift:$TAG_PLOT_ARM ${CACHE_IMAGE}:$TAG_PLOT_ARM
docker tag chaindrift:$TAG_CHAINAI_ARM ${CACHE_IMAGE}:$TAG_CHAINAI_ARM
docker tag chaindrift:$TAG_CHAINAI_RL_ARM ${CACHE_IMAGE}:$TAG_CHAINAI_RL_ARM

# Run backtest
docker run --rm -v $(pwd)/tests/testdata/config.tests.json:/chaindrift/config.json:ro -v $(pwd)/tests:/tests chaindrift:${TAG_ARM} backtesting --datadir /tests/testdata --strategy-path /tests/strategy/strats/ --strategy StrategyTestV3

if [ $? -ne 0 ]; then
    echo "failed running backtest"
    return 1
fi

docker images

docker push ${CACHE_IMAGE}:$TAG_PLOT_ARM
docker push ${CACHE_IMAGE}:$TAG_CHAINAI_ARM
docker push ${CACHE_IMAGE}:$TAG_CHAINAI_RL_ARM
docker push ${CACHE_IMAGE}:$TAG_ARM

# Create multi-arch image
# Make sure that all images contained here are pushed to github first.
# Otherwise installation might fail.
echo "create manifests"

docker manifest create ${IMAGE_NAME}:${TAG} ${CACHE_IMAGE}:${TAG} ${CACHE_IMAGE}:${TAG_ARM} ${IMAGE_NAME}:${TAG_PI}
docker manifest push -p ${IMAGE_NAME}:${TAG}

docker manifest create ${IMAGE_NAME}:${TAG_PLOT} ${CACHE_IMAGE}:${TAG_PLOT} ${CACHE_IMAGE}:${TAG_PLOT_ARM}
docker manifest push -p ${IMAGE_NAME}:${TAG_PLOT}

docker manifest create ${IMAGE_NAME}:${TAG_CHAINAI} ${CACHE_IMAGE}:${TAG_CHAINAI} ${CACHE_IMAGE}:${TAG_CHAINAI_ARM}
docker manifest push -p ${IMAGE_NAME}:${TAG_CHAINAI}

docker manifest create ${IMAGE_NAME}:${TAG_CHAINAI_RL} ${CACHE_IMAGE}:${TAG_CHAINAI_RL} ${CACHE_IMAGE}:${TAG_CHAINAI_RL_ARM}
docker manifest push -p ${IMAGE_NAME}:${TAG_CHAINAI_RL}

# Create special Torch tag - which is identical to the RL tag.
docker manifest create ${IMAGE_NAME}:${TAG_CHAINAI_TORCH} ${CACHE_IMAGE}:${TAG_CHAINAI_RL} ${CACHE_IMAGE}:${TAG_CHAINAI_RL_ARM}
docker manifest push -p ${IMAGE_NAME}:${TAG_CHAINAI_TORCH}

# copy images to ghcr.io

alias crane="docker run --rm -i -v $(pwd)/.crane:/home/nonroot/.docker/ gcr.io/go-containerregistry/crane"
mkdir .crane
chmod a+rwx .crane

echo "${GHCR_TOKEN}" | crane auth login ghcr.io -u "${GHCR_USERNAME}" --password-stdin

crane copy ${IMAGE_NAME}:${TAG_CHAINAI_RL} ${GHCR_IMAGE_NAME}:${TAG_CHAINAI_RL}
crane copy ${IMAGE_NAME}:${TAG_CHAINAI_RL} ${GHCR_IMAGE_NAME}:${TAG_CHAINAI_TORCH}
crane copy ${IMAGE_NAME}:${TAG_CHAINAI} ${GHCR_IMAGE_NAME}:${TAG_CHAINAI}
crane copy ${IMAGE_NAME}:${TAG_PLOT} ${GHCR_IMAGE_NAME}:${TAG_PLOT}
crane copy ${IMAGE_NAME}:${TAG} ${GHCR_IMAGE_NAME}:${TAG}

# Tag as latest for develop builds
if [ "${TAG}" = "develop" ]; then
    echo 'Tagging image as latest'
    docker manifest create ${IMAGE_NAME}:latest ${CACHE_IMAGE}:${TAG_ARM} ${IMAGE_NAME}:${TAG_PI} ${CACHE_IMAGE}:${TAG}
    docker manifest push -p ${IMAGE_NAME}:latest

    crane copy ${IMAGE_NAME}:latest ${GHCR_IMAGE_NAME}:latest
fi

docker images
rm -rf .crane

# Cleanup old images from arm64 node.
docker image prune -a --force --filter "until=24h"
