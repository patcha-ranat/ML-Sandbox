output "repository_url" {
  description = "The URL of the artifact registry repository"
  value       = "${var.region}-docker.pkg.dev/${var.project}/${google_artifact_registry_repository.n8n_repo.repository_id}"
}
