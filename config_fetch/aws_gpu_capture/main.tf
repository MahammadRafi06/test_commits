terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_ssm_parameter" "dlami" {
  name = var.ami_ssm_parameter
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

locals {
  subnet_id = var.subnet_id != "" ? var.subnet_id : sort(data.aws_subnets.default.ids)[0]
  common_tags = merge(
    var.tags,
    {
      Name      = var.name
      Component = "dynamo-config-capture"
    }
  )
}

resource "aws_key_pair" "capture" {
  key_name_prefix = "${var.name}-"
  public_key      = file(var.public_key_path)

  tags = local.common_tags
}

resource "aws_security_group" "capture" {
  name_prefix = "${var.name}-"
  description = "SSH access for Dynamo config capture"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from local workstation"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_instance" "capture" {
  ami                         = data.aws_ssm_parameter.dlami.value
  instance_type               = var.instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.capture.id]
  key_name                    = aws_key_pair.capture.key_name
  associate_public_ip_address = true

  instance_initiated_shutdown_behavior = "stop"

  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  lifecycle {
    ignore_changes = [
      associate_public_ip_address,
    ]
  }

  user_data = <<-EOF
    #!/usr/bin/env bash
    set -euxo pipefail
    if ! command -v docker >/dev/null 2>&1; then
      apt-get update
      apt-get install -y docker.io
    fi
    systemctl enable --now docker
    usermod -aG docker ubuntu || true
    mkdir -p /opt/dynamo-config-capture
  EOF

  tags = local.common_tags
}
