terraform {
  required_version = "~>1.12.2"
  required_providers {
    google = {
      version = "~>6.41.0"
      source  = "hashicorp/google"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
  backend "gcs" {
    bucket = "n8n_public_artifact_bucket"
    prefix = "terraform/state/n8n-public/02-backend-db"
  }
}

provider "google" {
  project = var.project
  region  = var.region
}

resource "random_password" "db_password" {
  length  = 16
  special = true
}

resource "google_sql_user" "n8n_user" {
  name     = var.db_username
  instance = google_sql_database_instance.n8n_backend_instance.name
  password = random_password.db_password.result
}

resource "google_sql_database_instance" "n8n_backend_instance" {
  name                = var.db_instance_name
  database_version    = "POSTGRES_16"
  region              = var.region
  deletion_protection = false
  root_password       = random_password.db_password.result

  settings {
    tier    = var.db_tier
    edition = "ENTERPRISE"

    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    disk_size         = 10

    # Enable public IP for public deployment
    ip_configuration {
      ipv4_enabled = true
      # For production, restrict authorized networks
      # For POC, allowing all (use with caution)
      authorized_networks {
        name  = "allow-all"
        value = "0.0.0.0/0"
      }
    }
  }
}

resource "google_sql_database" "n8n_backend_db" {
  name     = "n8n_db"
  instance = google_sql_database_instance.n8n_backend_instance.name
}
