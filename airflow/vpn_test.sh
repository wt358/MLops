#!/bin/sh

TAG_CUDA=$1
message=$2

docker images
docker build -t ctf-mlops.kr.ncr.ntruss.com/vpn:$TAG_CUDA -f ./Dockerfile/vpn-Dockerfile ./
docker push ctf-mlops.kr.ncr.ntruss.com/vpn:$TAG_CUDA
docker rmi ctf-mlops.kr.ncr.ntruss.com/vpn:$TAG_CUDA

git add .
git commit -m "$2"
git push origin main
