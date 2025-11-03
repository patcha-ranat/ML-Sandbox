output "bucket_name" {
  description = "The name of the GCS bucket for Terraform state"
  value       = google_storage_bucket.backend_bucket.name
}
