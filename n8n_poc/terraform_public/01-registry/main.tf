terraform {
  required_version = "~>1.12.2"
  required_providers {
    google = {
      version = "~>6.41.0"
      source  = "hashicorp/google"
    }
  }
  backend "gcs" {
    bucket = "n8n_public_artifact_bucket"
    prefix = "terraform/state/n8n-public/01-registry"
  }
}

provider "google" {
  project = var.project
  region  = var.region
}

resource "google_artifact_registry_repository" "n8n_repo" {
  location      = var.region
  repository_id = "n8n-image-repo"
  description   = "Image Registry for n8n"
  format        = "DOCKER"
  mode          = "STANDARD_REPOSITORY"
}
