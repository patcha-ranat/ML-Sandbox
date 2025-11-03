output "database_name" {
  description = "The name of the database"
  value       = google_sql_database.n8n_backend_db.name
}

output "database_username" {
  description = "The database username"
  value       = google_sql_user.n8n_user.name
}

output "database_password" {
  description = "The database password"
  value       = random_password.db_password.result
  sensitive   = true
}

output "instance_connection_name" {
  description = "The connection name of the Cloud SQL instance for Cloud SQL Proxy"
  value       = google_sql_database_instance.n8n_backend_instance.connection_name
}

output "instance_public_ip_address" {
  description = "The public IP address of the Cloud SQL instance"
  value       = google_sql_database_instance.n8n_backend_instance.public_ip_address
}
