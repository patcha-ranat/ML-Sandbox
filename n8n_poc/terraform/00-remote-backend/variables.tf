variable "project" {
  type      = string
  sensitive = true
}

variable "bucket_name" {
  type      = string
  sensitive = true
}

variable "region" {
  type    = string
  default = "asia-southeast1"
}
