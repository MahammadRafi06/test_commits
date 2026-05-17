output "instance_id" {
  value = aws_instance.capture.id
}

output "public_ip" {
  value = aws_instance.capture.public_ip
}

output "ssh_user" {
  value = "ubuntu"
}

output "region" {
  value = var.region
}

output "root_volume_size_gb" {
  value = var.root_volume_size_gb
}
