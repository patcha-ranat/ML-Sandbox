terraform {
  required_version = "~>1.12.2"
  required_providers {
    google = {
      version = "~>6.41.0"
      source  = "hashicorp/google"
    }
  }
}

provider "google" {
  project = var.project
  region  = var.region
}

resource "google_storage_bucket" "backend_bucket" {
  name          = var.bucket_name
  location      = var.region
  project       = var.project
  storage_class = "STANDARD"
  force_destroy = true
  versioning {
    enabled = true
  }
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}
