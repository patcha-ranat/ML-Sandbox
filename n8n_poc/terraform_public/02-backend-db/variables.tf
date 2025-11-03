variable "project" {
  type        = string
  default     = "your-gcp-project-name"
  description = "GCP Project ID"
}

variable "region" {
  type        = string
  default     = "asia-southeast1"
  description = "GCP region for resources"
}

variable "db_instance_name" {
  type        = string
  default     = "n8n-backend-instance"
  description = "Name of the Cloud SQL instance"
}

variable "db_username" {
  type        = string
  default     = "n8n_user"
  description = "Database username for n8n"
}

variable "db_tier" {
  type        = string
  default     = "db-f1-micro"
  description = "Database tier for Cloud SQL instance"
}
