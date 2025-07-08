terraform {
  required_version = "~>1.12.2"

  required_providers {
    google = {
      version = "~>6.41.0"
      source  = "hashicorp/google"
    }
  }

  backend "gcs" {
    bucket = "[REDACTED]" # change this to artifact bucket name
    prefix = "[REDACTED]" # terraform/state/n8n_XXX_poc/01_registry
  }
}

provider "google" {
  project = var.project
  region  = var.region
}

resource "google_artifact_registry_repository" "n8n_remote_repo" {
  location      = var.region
  repository_id = "n8n-remote-repo"
  description   = "Private Image Registry for n8n"
  format        = "DOCKER"
  mode          = "STANDARD"
  # mode          = "REMOTE_REPOSITORY"
  # remote_repository_config {
  #   description = "docker hub"
  #   docker_repository {
  #     public_repository = "DOCKER_HUB"
  #   }
  # }
}