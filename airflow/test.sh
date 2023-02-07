#!/bin/sh

TAG=$1
message=$2

docker images
docker build -t ctf-mlops.kr.ncr.ntruss.com/cuda:$TAG -f ./Dockerfile/gpu-Dockerfile ./Dockerfile/
docker push ctf-mlops.kr.ncr.ntruss.com/cuda:$TAG

git add .
git commit -m "$2"
git push origin main
