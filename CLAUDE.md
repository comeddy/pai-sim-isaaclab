# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physical AI 워크샵 프로젝트 — Terraform으로 AWS GPU 인스턴스(g6e.4xlarge, NVIDIA L40S)를 프로비저닝하고, NVIDIA Isaac Lab에서 ANYmal-C 4족 보행 로봇의 강화학습(PPO)을 실행하는 end-to-end 파이프라인. 프로젝트 언어는 한국어.

## Commands

### Terraform (인프라)

```bash
terraform init          # 프로바이더 초기화
terraform plan          # 변경 사항 미리보기
terraform apply         # 인프라 배포 (~3분)
terraform destroy       # 전체 리소스 삭제
terraform validate      # HCL 구문 검증
terraform fmt           # HCL 포맷팅 (자동 정렬)
terraform fmt -check    # 포맷 검사 (CI용)
```

### GitBook 워크샵 (workshop/)

```bash
cd workshop
npm install             # 의존성 설치 (honkit)
npx honkit serve        # http://localhost:4000 로컬 서버
npx honkit build        # _book/ 정적 빌드
```

## Architecture

### Terraform 레이어 (루트 디렉토리)

단일 Terraform 구성으로, 모듈 분리 없이 `main.tf` 하나에 전체 인프라를 정의:

- **Networking**: VPC → Internet Gateway → Public Subnet → Route Table
- **Compute**: g6e.4xlarge EC2 (Deep Learning Base AMI, Ubuntu 22.04)
- **Storage**: Root EBS (gp3, Docker 이미지용) + Data EBS (gp3, 체크포인트/데이터셋) + Instance Store NVMe (셰이더 캐시/scratch)
- **IAM**: EC2 역할 → S3 체크포인트 접근 + SSM Session Manager
- **Monitoring**: CloudWatch 알람 → GPU 30분 idle 시 자동 Stop

`variables.tf`에 모든 입력 변수, `outputs.tf`에 SSH 명령어/인스턴스 정보 출력.

### user_data.sh — EC2 부트스트랩

Terraform `templatefile()`로 렌더링되는 셸 스크립트. 주요 단계:
1. dpkg lock 경합 해결 후 시스템 패키지 설치
2. Data EBS 볼륨 마운트 (Nitro `/dev/nvme*n1` 동적 탐색)
3. Instance Store NVMe 마운트 (또는 `/opt/dlami/nvme` 재활용)
4. NGC 로그인 + Isaac Sim Docker 이미지 pull
5. Isaac Lab Docker 빌드 (소스 빌드 fallback)
6. Wrapper 스크립트 생성: `isaac-lab-run`, `isaac-sim-shell`, `sync-checkpoints`
7. GPU 메트릭 CloudWatch 전송 cron (5분 간격)

**중요**: `user_data.sh` 내에서 Terraform 변수는 `${var_name}`, 셸 변수는 `$${SHELL_VAR}` (double dollar 이스케이프). 이를 혼동하면 templatefile 렌더링이 실패한다.

### 워크샵 (workshop/)

HonKit(GitBook fork) 기반 7개 Lab + 3개 Appendix 한국어 문서. `SUMMARY.md`가 목차, `book.json`이 설정, `chapters/`에 마크다운 콘텐츠. `assets/`에 스크린샷.

### 산출물

- `models/`: 학습된 정책 (policy_jit.pt, policy.onnx)
- `videos/`: Play 모드 녹화 MP4
- `images/`: 대시보드 스크린샷, 프레임 캡처
- `isaac_lab_dashboard.html`: Chart.js 기반 훈련 메트릭 대시보드

## Key Gotchas

- Isaac Lab NGC 이미지가 공식 제공되지 않으므로 소스에서 `docker compose --profile base build` 필요
- 빌드 후 코어 `isaaclab` 패키지가 누락됨 → `pip install --no-build-isolation -e source/isaaclab` 수동 설치 필수
- Docker 기본 entrypoint(`runheadless.sh`)는 스트리밍 서버 모드 → 훈련 시 반드시 `--entrypoint /workspace/isaaclab/isaaclab.sh` 오버라이드
- `terraform.tfvars`는 `.gitignore` 대상이 아님 — NGC API Key 등 민감 정보 커밋 주의
- `ngc_api_key` 변수는 `sensitive = true`로 설정됨
