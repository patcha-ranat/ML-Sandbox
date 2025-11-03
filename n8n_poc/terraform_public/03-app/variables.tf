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

variable "service_name" {
  type        = string
  default     = "n8n-service"
  description = "Name of the Cloud Run service"
}

variable "n8n_version" {
  type        = string
  default     = "1.100.0"
  description = "n8n version to deploy"
}

variable "n8n_host" {
  type        = string
  default     = "ntt-data"
  description = "Hostname for n8n service (e.g., your-domain.com)"
}
