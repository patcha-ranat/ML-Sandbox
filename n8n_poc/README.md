# n8n Docker Local

*Patcharanat P.*

Only focus on n8n deployment

## Local Development

```bash
docker compose up
```

## Cloud Deployment

```bash
# Initial GCS for storing remote backend
terraform -chdir="terraform/00-remote-backend" init
# terraform fmt
# terraform plan
terraform -chdir="terraform/00-remote-backend" plan -var-file="secret.tfvars"
terraform -chdir="terraform/00-remote-backend" apply -var-file="secret.tfvars"
# terraform -chdir="terraform/00-remote-backend" destroy -var-file="secret.tfvars"

# Artifact Registry - separated for automation benefits (pull-push image between steps)
terraform -chdir="terraform/01-registry" init
terraform -chdir="terraform/01-registry" plan -var-file="secret.tfvars"
terraform -chdir="terraform/01-registry" apply -var-file="secret.tfvars"
# terraform -chdir="terraform/01-registry" destroy -var-file="secret.tfvars"

# Cloud SQL - long create time
terraform -chdir="terraform/02-backend-db" init
terraform -chdir="terraform/02-backend-db" plan -var-file="secret.tfvars"
terraform -chdir="terraform/02-backend-db" apply -var-file="secret.tfvars"
# terraform -chdir="terraform/02-backend-db" destroy -var-file="secret.tfvars"

# Cloud Run - deploy/undeploy service app
terraform -chdir="terraform/03-app" init
terraform -chdir="terraform/03-app" plan -var-file="secret.tfvars"
terraform -chdir="terraform/03-app" apply -var-file="secret.tfvars"
terraform -chdir="terraform/03-app" destroy -var-file="secret.tfvars"
```

References:
- n8n
    - [n8n - Hosting with docker](https://docs.n8n.io/hosting/installation/docker/)
    - [n8n - Hosting with docker compose](https://docs.n8n.io/hosting/installation/server-setups/docker-compose/)
    - [GitHub n8n - docker compose with postgres example](https://github.com/n8n-io/n8n-hosting/blob/main/docker-compose/withPostgres/README.md)
    - [GitHub n8n](https://github.com/n8n-io/n8n?tab=readme-ov-file)
    - [Google: OAuth2 single service](https://docs.n8n.io/integrations/builtin/credentials/google/oauth-single-service/?utm_source=n8n_app&utm_medium=credential_settings&utm_campaign=create_new_credentials_modal)
- Gemini
    - [Model & Pricing](https://ai.google.dev/gemini-api/docs/pricing)
    - [Getting API Key](https://aistudio.google.com/u/1/apikey)
- Google Authentication Troubleshooting
    - [Google Cloud OAuth Authorization Error: This client is restricted to users within its organization](https://stackoverflow.com/questions/55285369/google-cloud-oauth-authorization-error-this-client-is-restricted-to-users-withi)
- Terraform
    - [cloud Run Provisioning](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/cloud_run_v2_service)
    - [Artifact Provisioning](https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/artifact_registry_repository)