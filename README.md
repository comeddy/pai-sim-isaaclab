# Isaac Lab Headless Training — Terraform on AWS g6e.4xlarge

NVIDIA L40S GPU 기반 Isaac Lab 헤드리스 강화학습 환경을 AWS에 자동 프로비저닝하는 Terraform 구성입니다.

## 인프라 구성

```
┌─────────────────────────────────────────────────────────────┐
│  VPC 10.0.0.0/16                                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Public Subnet 10.0.1.0/24                             │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  g6e.4xlarge                                     │  │  │
│  │  │  ┌────────────┐  ┌───────────────────────────┐   │  │  │
│  │  │  │ L40S 48GB  │  │ Docker                    │   │  │  │
│  │  │  │ GPU        │  │ ┌─────────────────────┐   │   │  │  │
│  │  │  └────────────┘  │ │ Isaac Lab Container  │   │   │  │  │
│  │  │                  │ │ (headless training)  │   │   │  │  │
│  │  │  16 vCPU         │ └─────────────────────┘   │   │  │  │
│  │  │  128 GiB RAM     └───────────────────────────┘   │  │  │
│  │  │                                                  │  │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │  │  │
│  │  │  │ Root EBS │  │ Data EBS │  │ NVMe 600GB    │  │  │  │
│  │  │  │ 300GB    │  │ 500GB    │  │ (scratch)     │  │  │  │
│  │  │  │ gp3      │  │ gp3      │  │ instance store│  │  │  │
│  │  │  └──────────┘  └──────────┘  └───────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────────┘
                  │
          ┌───────┴───────┐
          │  S3 Bucket    │  ← 체크포인트 자동 동기화 (30분마다)
          │  (checkpoints)│
          └───────────────┘
```

## 사전 준비

1. **Terraform** >= 1.5 설치
2. **AWS CLI** 구성 (`aws configure`)
3. **NVIDIA NGC API Key** 발급: https://ngc.nvidia.com/setup/api-key
4. **S3 버킷** 생성 (체크포인트 저장용)

## 빠른 시작

```bash
# 1. 변수 파일 생성
cp terraform.tfvars.example terraform.tfvars
# terraform.tfvars 편집 — NGC API Key, SSH 키, S3 버킷 등 입력

# 2. 초기화 & 배포
terraform init
terraform plan
terraform apply

# 3. SSH 접속
# (자동 생성 키 사용 시)
terraform output -raw ssh_private_key > isaac-lab-key
chmod 600 isaac-lab-key
ssh -i isaac-lab-key ubuntu@$(terraform output -raw public_ip)

# 4. 학습 실행
isaac-lab-run source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Anymal-C-v0 --headless

# 5. 정리
terraform destroy
```

## 인스턴스 스펙

| 항목 | 값 |
|------|-----|
| 인스턴스 | g6e.4xlarge |
| GPU | 1× NVIDIA L40S (48 GB VRAM) |
| vCPU | 16 (AMD EPYC 3세대) |
| RAM | 128 GiB |
| NVMe | 600 GB (인스턴스 스토어) |
| 네트워크 | 20 Gbps |
| 비용 | ~$3.00/hr (on-demand) |

## 스토리지 레이아웃

| 마운트 | 용도 | 유형 |
|--------|------|------|
| `/` (root) | OS, Docker 이미지 | EBS gp3 300GB |
| `/data` | 데이터셋, 체크포인트, 로그 | EBS gp3 500GB |
| `/scratch` | 셰이더 캐시, 임시 파일 | NVMe 600GB |

## 내장 도구

- **`isaac-lab-run <script> [args]`** — Isaac Lab 컨테이너에서 헤드리스 학습 실행
- **`isaac-sim-shell`** — Isaac Sim 컨테이너 대화형 셸
- **`sync-checkpoints`** — S3로 체크포인트 수동 동기화

## 비용 절감 팁

1. **GPU Idle Stop**: `enable_idle_stop = true`로 GPU 30분 유휴 시 인스턴스 자동 중지
2. **Spot Instance**: `main.tf`의 spot 블록 주석 해제 시 ~60-70% 절감 가능
3. **서울 리전**: `aws_region = "ap-northeast-2"` — g6e.4xlarge 사용 가능

## 커스터마이징

### 다른 RL 프레임워크 사용

```bash
# RSL-RL (기본)
isaac-lab-run source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Anymal-C-v0 --headless

# Stable Baselines3
isaac-lab-run source/standalone/workflows/sb3/train.py \
  --task Isaac-Velocity-Flat-Anymal-C-v0 --headless

# SKRL
isaac-lab-run source/standalone/workflows/skrl/train.py \
  --task Isaac-Velocity-Rough-Anymal-C-v0 --headless
```

### Multi-GPU (스케일업)

`g6e.12xlarge` (4× L40S) 또는 `g6e.48xlarge` (8× L40S)로 변경하려면
`main.tf`의 `instance_type`만 수정하면 됩니다.
