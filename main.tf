###############################################################################
# Isaac Lab Headless Training Environment — g6e.4xlarge (NVIDIA L40S)
# Terraform >= 1.5 | AWS Provider >= 5.0
###############################################################################

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = ">= 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --------------------------------------------------------------------------- #
# Data Sources
# --------------------------------------------------------------------------- #

# Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04) — pre-installed
# NVIDIA driver, CUDA, Docker, NVIDIA Container Toolkit
data "aws_ami" "dl_base" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

# --------------------------------------------------------------------------- #
# Networking
# --------------------------------------------------------------------------- #

resource "aws_vpc" "isaac" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name    = "${var.project_name}-vpc"
    Project = var.project_name
  }
}

resource "aws_internet_gateway" "isaac" {
  vpc_id = aws_vpc.isaac.id

  tags = {
    Name    = "${var.project_name}-igw"
    Project = var.project_name
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.isaac.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name    = "${var.project_name}-public-subnet"
    Project = var.project_name
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.isaac.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.isaac.id
  }

  tags = {
    Name    = "${var.project_name}-public-rt"
    Project = var.project_name
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# --------------------------------------------------------------------------- #
# Security Group
# --------------------------------------------------------------------------- #

resource "aws_security_group" "isaac" {
  name_prefix = "${var.project_name}-sg-"
  description = "Isaac Lab training instance security group"
  vpc_id      = aws_vpc.isaac.id

  # SSH
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # All outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-sg"
    Project = var.project_name
  }

  lifecycle {
    create_before_destroy = true
  }
}

# --------------------------------------------------------------------------- #
# SSH Key Pair
# --------------------------------------------------------------------------- #

# Option 1: Use an existing AWS key pair by name
data "aws_key_pair" "existing" {
  count    = var.existing_key_name != "" ? 1 : 0
  key_name = var.existing_key_name
}

# Option 2: Auto-generate a key pair (only when no existing key and no public key provided)
resource "tls_private_key" "ssh" {
  count     = var.existing_key_name == "" && var.ssh_public_key == "" ? 1 : 0
  algorithm = "ED25519"
}

resource "aws_key_pair" "isaac" {
  count      = var.existing_key_name == "" ? 1 : 0
  key_name   = "${var.project_name}-key"
  public_key = var.ssh_public_key != "" ? var.ssh_public_key : tls_private_key.ssh[0].public_key_openssh

  tags = {
    Project = var.project_name
  }
}

locals {
  key_name = var.existing_key_name != "" ? var.existing_key_name : aws_key_pair.isaac[0].key_name
}

# --------------------------------------------------------------------------- #
# IAM Role — S3 access for checkpoints + SSM Session Manager
# --------------------------------------------------------------------------- #

resource "aws_iam_role" "isaac" {
  name_prefix = "${var.project_name}-role-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })

  tags = {
    Project = var.project_name
  }
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.isaac.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "s3_checkpoints" {
  name_prefix = "s3-checkpoints-"
  role        = aws_iam_role.isaac.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
      Resource = [
        "arn:aws:s3:::${var.checkpoint_bucket}",
        "arn:aws:s3:::${var.checkpoint_bucket}/*"
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "isaac" {
  name_prefix = "${var.project_name}-profile-"
  role        = aws_iam_role.isaac.name

  tags = {
    Project = var.project_name
  }
}

# --------------------------------------------------------------------------- #
# EBS Volume for datasets / checkpoints
# --------------------------------------------------------------------------- #

resource "aws_ebs_volume" "data" {
  availability_zone = data.aws_availability_zones.available.names[0]
  size              = var.data_volume_size_gb
  type              = "gp3"
  throughput        = 500
  iops              = 6000
  encrypted         = true

  tags = {
    Name    = "${var.project_name}-data"
    Project = var.project_name
  }
}

# --------------------------------------------------------------------------- #
# EC2 Instance — g6e.4xlarge (1x NVIDIA L40S, 16 vCPU, 128 GiB RAM)
# --------------------------------------------------------------------------- #

resource "aws_instance" "isaac" {
  ami                    = data.aws_ami.dl_base.id
  instance_type          = "g6e.4xlarge"
  key_name               = local.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.isaac.id]
  iam_instance_profile   = aws_iam_instance_profile.isaac.name

  # Root volume — OS + Docker images
  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    throughput            = 500
    iops                  = 6000
    encrypted             = true
    delete_on_termination = true
  }

  # Instance store (600 GB NVMe) is auto-attached on g6e.4xlarge

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"   # IMDSv2 only
    http_put_response_hop_limit = 2
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    isaac_lab_version = var.isaac_lab_version
    isaac_sim_version = var.isaac_sim_version
    ngc_api_key       = var.ngc_api_key
    checkpoint_bucket = var.checkpoint_bucket
  }))

  tags = {
    Name    = "${var.project_name}-gpu"
    Project = var.project_name
  }

  # Spot 인스턴스를 원하면 아래 블록 주석 해제
  # instance_market_options {
  #   market_type = "spot"
  #   spot_options {
  #     max_price          = var.spot_max_price
  #     spot_instance_type = "persistent"
  #   }
  # }

  lifecycle {
    ignore_changes = [ami]
  }
}

resource "aws_volume_attachment" "data" {
  device_name = "/dev/xvdf"
  volume_id   = aws_ebs_volume.data.id
  instance_id = aws_instance.isaac.id
}

# --------------------------------------------------------------------------- #
# CloudWatch Alarm — GPU utilization (optional auto-stop)
# --------------------------------------------------------------------------- #

resource "aws_cloudwatch_metric_alarm" "gpu_idle" {
  count               = var.enable_idle_stop ? 1 : 0
  alarm_name          = "${var.project_name}-gpu-idle-stop"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 6           # 30 min (6 × 5 min)
  metric_name         = "GPUUtilization"
  namespace           = "CWAgent"
  period              = 300
  statistic           = "Average"
  threshold           = 5
  alarm_description   = "Stop instance when GPU idle for 30 min"

  alarm_actions = [
    "arn:aws:automate:${var.aws_region}:ec2:stop"
  ]

  dimensions = {
    InstanceId = aws_instance.isaac.id
  }

  tags = {
    Project = var.project_name
  }
}
