variable "region" {
  description = "AWS region for the GPU capture instance."
  type        = string
  default     = "us-west-2"
}

variable "name" {
  description = "Name prefix for AWS resources."
  type        = string
  default     = "dynamo-config-capture"
}

variable "instance_type" {
  description = "GPU instance type."
  type        = string
  default     = "g4dn.xlarge"
}

variable "root_volume_size_gb" {
  description = "Root EBS volume size. Dynamo runtime images are large, so keep this generous."
  type        = number
  default     = 300
}

variable "ami_ssm_parameter" {
  description = "Public SSM parameter for the Deep Learning AMI."
  type        = string
  default     = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
}

variable "subnet_id" {
  description = "Optional subnet ID. Defaults to the first default subnet in the region."
  type        = string
  default     = ""
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH to the capture instance."
  type        = string
}

variable "public_key_path" {
  description = "Path to the SSH public key to register with EC2."
  type        = string
}

variable "tags" {
  description = "Extra tags for created AWS resources."
  type        = map(string)
  default     = {}
}
