# n8n Docker Local and Self Hosting with Serverless

*Patcharanat P.*

## Table of Contents
1. [Getting Started Local](#1-getting-started-local)
2. [Getting Started Cloud](#2-getting-started-cloud)
3. [References](#3-references)

## 1. Getting Started Local
```bash
# create .env file to use with docker-compose.yml
cp local/.env.example local/.env

# start
docker compose -f local/docker-compose.yml up --build

# access the app via: http://localhost:5678

# stop
docker compose -f local/docker-compose.yml down -v
```

## 2. Getting Started Cloud
```bash
# init
cd 00-remote-backend && \
    terraform init && \
    terraform plan && \
    terraform apply -auto-approve && \
    cd ..

cd 01-registry && \
    terraform init && \
    cd ..

cd 02-backend-db && \
    terraform init && \
    cd ..

cd 03-app && \
    terraform init && \
    cd ..

# provision
# terraform -chdir=00-remote-backend apply -auto-approve
terraform -chdir=01-registry apply -auto-approve
terraform -chdir=02-backend-db apply -auto-approve

# pre-requisite before deploying app
export N8N_IMAGE_VERSION="1.100.0"
export GCP_REGION="asia-southeast1"
export GCP_PROJECT="your-gcp-project-name"
export N8N_PRIVATE_REGISTRY="n8n-image-repo"

gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

docker pull docker.n8n.io/n8nio/n8n:${N8N_IMAGE_VERSION}

docker tag docker.n8n.io/n8nio/n8n:${N8N_IMAGE_VERSION} ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${N8N_PRIVATE_REGISTRY}/n8nio/n8n:${N8N_IMAGE_VERSION}

docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${N8N_PRIVATE_REGISTRY}/n8nio/n8n:${N8N_IMAGE_VERSION}

# deploy app
terraform -chdir=03-app apply -auto-approve

# access the app via the link terraform provided after provisioning

# de-provision
terraform -chdir=03-app destroy -auto-approve
terraform -chdir=02-backend-db destroy -auto-approve
terraform -chdir=01-registry destroy -auto-approve
# terraform -chdir=00-remote-backend destroy -auto-approve
```

## 3. References
- n8n
    - [n8n - Hosting with docker](https://docs.n8n.io/hosting/installation/docker/)
    - [n8n - Hosting with docker compose](https://docs.n8n.io/hosting/installation/server-setups/docker-compose/)
    - [GitHub n8n - docker compose with postgres example](https://github.com/n8n-io/n8n-hosting/blob/main/docker-compose/withPostgres/README.md)
    - [GitHub n8n](https://github.com/n8n-io/n8n?tab=readme-ov-file)
    - [Google: OAuth2 single service](https://docs.n8n.io/integrations/builtin/credentials/google/oauth-single-service/)
    - [Hosting n8n on Google Cloud Run](https://docs.n8n.io/hosting/installation/server-setups/google-cloud-run/)
- Gemini
    - [Model & Pricing](https://ai.google.dev/gemini-api/docs/pricing)
    - [Getting API Key](https://aistudio.google.com/u/1/apikey)
- Google Authentication Troubleshooting
    - [Google Cloud OAuth Authorization Error: This client is restricted to users within its organization](https://stackoverflow.com/questions/55285369/google-cloud-oauth-authorization-error-this-client-is-restricted-to-users-withi)
- Terraform
    - [cloud Run Provisioning](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/cloud_run_v2_service)
    - [Artifact Provisioning](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/artifact_registry_repository)