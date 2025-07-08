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
    prefix = "[REDACTED]" # terraform/state/n8n_XXX_poc/02_backend_db
  }
}

provider "google" {
  project = var.project
  region  = var.region
}

resource "google_sql_database_instance" "n8n_backend_instance" {
  name                = "n8n-backend-instance"
  database_version    = "POSTGRES_16"
  region              = var.region
  deletion_protection = false

  settings {
    tier    = "db-f1-micro"
    edition = "ENTERPRISE"
  }
}

resource "google_sql_database" "n8n_backend_db" {
  name     = "n8n-backend"
  instance = google_sql_database_instance.n8n_backend_instance.name
}