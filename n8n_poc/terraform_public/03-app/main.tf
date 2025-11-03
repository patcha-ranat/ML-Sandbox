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
    prefix = "terraform/state/n8n-public/03-app"
  }
}

provider "google" {
  project = var.project
  region  = var.region
}

# Reference artifact registry state
data "terraform_remote_state" "artifact_registry" {
  backend = "gcs"
  config = {
    bucket = "n8n_public_artifact_bucket"
    prefix = "terraform/state/n8n-public/01-registry"
  }
}

# Reference database state
data "terraform_remote_state" "database" {
  backend = "gcs"
  config = {
    bucket = "n8n_public_artifact_bucket"
    prefix = "terraform/state/n8n-public/02-backend-db"
  }
}

# Secret Manager for database password
resource "google_secret_manager_secret" "db_password" {
  secret_id = "n8n-db-password"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password_version" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = data.terraform_remote_state.database.outputs.database_password
}

# Service account for Cloud Run
resource "google_service_account" "n8n_service_account" {
  account_id   = "n8n-service-account"
  display_name = "n8n Service Account"
  description  = "Service account for n8n Cloud Run service"
}

# IAM permissions for service account
resource "google_project_iam_member" "n8n_secret_accessor" {
  project = var.project
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.n8n_service_account.email}"
}

resource "google_storage_bucket_iam_member" "n8n_data_access" {
  bucket = google_storage_bucket.n8n_data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.n8n_service_account.email}"
}

# Storage bucket for n8n data persistence - POC optimized (lowest cost)
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "n8n_data" {
  name          = "${var.project}-n8n-data-${random_id.bucket_suffix.hex}"
  location      = var.region
  force_destroy = true

  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
}

# n8n Application on Cloud run
resource "google_cloud_run_v2_service" "n8n_container" {
  name                = var.service_name
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    service_account = google_service_account.n8n_service_account.email

    # n8n Container
    containers {
      name  = "n8n-service"
      image = "${data.terraform_remote_state.artifact_registry.outputs.repository_url}/n8nio/n8n:${var.n8n_version}"

      ports {
        container_port = 5678
      }

      resources {
        limits = {
          cpu    = "1000m"
          memory = "1Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = false
      }

      # Environment variables for database connection
      env {
        name  = "DB_TYPE"
        value = "postgresdb"
      }
      env {
        name  = "DB_POSTGRESDB_HOST"
        value = data.terraform_remote_state.database.outputs.instance_public_ip_address
      }
      env {
        name  = "DB_POSTGRESDB_PORT"
        value = "5432"
      }
      env {
        name  = "DB_POSTGRESDB_DATABASE"
        value = data.terraform_remote_state.database.outputs.database_name
      }
      env {
        name  = "DB_POSTGRESDB_USER"
        value = data.terraform_remote_state.database.outputs.database_username
      }
      env {
        name = "DB_POSTGRESDB_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_password.secret_id
            version = "latest"
          }
        }
      }
      env {
        name  = "N8N_HOST"
        value = var.n8n_host
      }
      env {
        name  = "N8N_PORT"
        value = "5678"
      }
      env {
        name  = "N8N_PROTOCOL"
        value = "https"
      }
      env {
        name  = "WEBHOOK_URL"
        value = "https://${var.n8n_host}"
      }
      env {
        name  = "GENERIC_TIMEZONE"
        value = "Asia/Bangkok"
      }
      env {
        name  = "N8N_LOG_LEVEL"
        value = "info"
      }

      # Persistent volume for n8n data inside container
      volume_mounts {
        name       = "n8n-data"
        mount_path = "/home/node/.n8n"
      }

      # Health checks
      startup_probe {
        http_get {
          path = "/healthz"
          port = 5678
        }
        initial_delay_seconds = 30
        timeout_seconds       = 10
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/healthz"
          port = 5678
        }
        initial_delay_seconds = 30
        timeout_seconds       = 10
        period_seconds        = 30
        failure_threshold     = 3
      }
    }

    # Persistent volume for n8n data on cloud
    volumes {
      name = "n8n-data"
      gcs {
        bucket    = google_storage_bucket.n8n_data.name
        read_only = false
      }
    }
  }

  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }

  depends_on = [
    google_secret_manager_secret_version.db_password_version,
    google_project_iam_member.n8n_secret_accessor
  ]
}

# Allow unauthenticated access to Cloud Run
resource "google_cloud_run_service_iam_member" "public_access" {
  location = google_cloud_run_v2_service.n8n_container.location
  project  = google_cloud_run_v2_service.n8n_container.project
  service  = google_cloud_run_v2_service.n8n_container.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
