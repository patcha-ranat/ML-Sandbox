output "service_url" {
  description = "URL of the deployed n8n Cloud Run service"
  value       = google_cloud_run_v2_service.n8n_container.uri
}

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_v2_service.n8n_container.name
}

output "service_account_email" {
  description = "Email of the service account used by Cloud Run"
  value       = google_service_account.n8n_service_account.email
}

output "storage_bucket_name" {
  description = "Name of the GCS bucket used for n8n data persistence"
  value       = google_storage_bucket.n8n_data.name
}
