# Appendix A: 실전 트러블슈팅 12선

> ⚠️ <b>WARNING</b>
>
> 이 목록은 2026년 4월 실제 AWS 배포에서 겪은 문제들입니다. 공식 문서에 없는 실전 함정과 해결법을 담았습니다.

---

## 인프라 (Terraform / EC2)

### 1. dpkg Lock 경합

| | |
|---|---|
| <b>증상</b> | `user_data.sh`에서 `apt-get install` 실패 |
| <b>에러</b> | `E: Could not get lock /var/lib/dpkg/lock-frontend` |
| <b>원인</b> | Ubuntu `unattended-upgrades`가 부팅 직후 dpkg lock 점유 |
| <b>빈도</b> | 거의 매번 (3/3 시도에서 발생) |

```bash
# 해결: user_data.sh 상단에 추가
systemctl stop unattended-upgrades 2>/dev/null || true
systemctl disable unattended-upgrades 2>/dev/null || true
killall -q apt-get dpkg 2>/dev/null || true
sleep 5

# 모든 apt-get에 timeout 옵션
apt-get install -y -o DPkg::Lock::Timeout=120 ...
```

---

### 2. EBS 디바이스 이름 (Nitro)

| | |
|---|---|
| <b>증상</b> | `/dev/xvdf` 디바이스가 없음 |
| <b>원인</b> | g6e = Nitro 기반 → EBS가 `/dev/nvme*n1`으로 표시 |
| <b>해결</b> | 동적 디바이스 탐색 함수 사용 |

```bash
find_data_device() {
  for dev in /dev/nvme*n1; do
    [ -b "$dev" ] || continue
    # Root, LVM, 이미 마운트된 것 제외
    if lsblk -no MOUNTPOINT "$dev" 2>/dev/null | grep -q "^/$"; then continue; fi
    if lsblk -no MOUNTPOINT "$dev" 2>/dev/null | grep -q .; then continue; fi
    if pvs "$dev" 2>/dev/null | grep -q "$dev"; then continue; fi
    echo "$dev"; return
  done
}
```

---

### 3. DL AMI Instance Store 이미 마운트

| | |
|---|---|
| <b>증상</b> | `mkfs.ext4 /dev/nvmeXn1` → "device is in use" |
| <b>원인</b> | DL Base AMI가 NVMe instance store를 LVM으로 `/opt/dlami/nvme`에 마운트 |
| <b>해결</b> | 기존 마운트 재활용 |

```bash
if mountpoint -q /opt/dlami/nvme 2>/dev/null; then
  mkdir -p /opt/dlami/nvme/scratch
  ln -sfn /opt/dlami/nvme/scratch /scratch
else
  mkfs.ext4 -L scratch "$NVME_DEV"
  mkdir -p /scratch && mount "$NVME_DEV" /scratch
fi
```

---

### 4. Terraform templatefile 충돌

| | |
|---|---|
| <b>증상</b> | `terraform apply` 시 `Invalid reference` 에러 |
| <b>원인</b> | Bash `${VAR:-default}`가 Terraform 변수로 해석 |
| <b>해결</b> | `$${VAR:-default}`로 이스케이프 (double dollar) |

```hcl
# main.tf
user_data = base64encode(templatefile("user_data.sh", {
  ngc_api_key = var.ngc_api_key
  # Bash 변수는 $${HOME} 형태로 자동 이스케이프
}))
```

---

### 5. user\_data 재실행 안 됨

| | |
|---|---|
| <b>증상</b> | `user_data.sh` 수정 → `terraform apply` → 변화 없음 |
| <b>원인</b> | cloud-init은 첫 부팅에서만 `user_data` 실행 |
| <b>해결</b> | 인스턴스 재생성 |

```bash
terraform taint aws_instance.isaac
terraform apply
```

---

## Docker / Isaac Lab

### 6. Isaac Lab NGC 이미지 없음

| | |
|---|---|
| <b>증상</b> | `docker pull nvcr.io/nvidia/isaac-lab:v2.1.0` → Not Found |
| <b>원인</b> | Isaac Lab은 NGC에 pre-built 이미지를 <b>제공하지 않음</b> |
| <b>해결</b> | 소스에서 빌드 |

```bash
git clone --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && docker compose --profile base build
```

---

### 7. 코어 isaaclab 패키지 누락 ★

| | |
|---|---|
| <b>증상</b> | `ModuleNotFoundError: No module named 'isaaclab'` |
| <b>원인</b> | Docker build가 extension만 설치, 코어 패키지 누락 |
| <b>빈도</b> | 100% 발생 (가장 치명적) |
| <b>해결</b> | 수동 설치 후 docker commit |

```bash
docker run --name setup --gpus all --entrypoint bash isaac-lab-base:latest -c '
  /workspace/isaaclab/_isaac_sim/python.sh -m pip install \
    --no-build-isolation -e /workspace/isaaclab/source/isaaclab
'
docker commit setup isaac-lab-ready:latest
```

---

### 8. Docker Entrypoint 스트리밍 모드

| | |
|---|---|
| <b>증상</b> | 컨테이너 실행 후 훈련 미시작, GPU 유휴 |
| <b>원인</b> | 기본 ENTRYPOINT `runheadless.sh` = 스트리밍 서버 모드 |
| <b>해결</b> | entrypoint 오버라이드 |

```bash
docker run ... \
  --entrypoint /workspace/isaaclab/isaaclab.sh \
  isaac-lab-ready:latest \
  -p scripts/.../train.py --headless
```

---

### 9. 훈련 스크립트 경로 변경

| | |
|---|---|
| <b>증상</b> | `FileNotFoundError: source/standalone/workflows/rsl_rl/train.py` |
| <b>원인</b> | v2.1.0에서 경로 변경 |
| <b>해결</b> | 새 경로 사용 |

```
구: source/standalone/workflows/rsl_rl/train.py
신: scripts/reinforcement_learning/rsl_rl/train.py
```

---

### 10. setuptools 빌드 격리 문제

| | |
|---|---|
| <b>증상</b> | `pip install -e source/isaaclab` → `No module named 'pkg_resources'` |
| <b>원인</b> | pip 빌드 격리 환경에 setuptools 없음 |
| <b>해결</b> | `--no-build-isolation` 플래그 |

```bash
pip install --no-build-isolation -e /workspace/isaaclab/source/isaaclab
```

---

### 11. Volume Mount가 Editable Install 덮어쓰기

| | |
|---|---|
| <b>증상</b> | 패키지 설치 후에도 `ModuleNotFoundError` 지속 |
| <b>원인</b> | `-v host/source:/workspace/isaaclab/source:rw`가 `.pth` 참조 파괴 |
| <b>해결</b> | editable install된 소스 디렉토리를 마운트하지 않기 |

```bash
# ❌ 하지 마세요
-v "/opt/isaaclab/source:/workspace/isaaclab/source:rw"

# ✅ 대신 데이터만 마운트
-v "/data/checkpoints:/workspace/isaaclab/logs:rw"
```

---

### 12. 셰이더 캐시 첫 실행 지연

| | |
|---|---|
| <b>증상</b> | 훈련 시작 후 4분간 아무 출력 없음 (행 걸린 줄 착각) |
| <b>원인</b> | Isaac Sim의 Vulkan RtPso 셰이더 파이프라인 첫 컴파일 |
| <b>해결</b> | 인내심 + 캐시 볼륨 마운트로 재컴파일 방지 |

```bash
-v "$CACHE_DIR/kit:/isaac-sim/kit/cache:rw"
```

---

## 빠른 참조 표

| # | 함정 | 카테고리 | 심각도 |
|---|------|---------|--------|
| 1 | dpkg lock | 인프라 | 🟡 중 |
| 2 | EBS 디바이스 | 인프라 | 🟡 중 |
| 3 | Instance store | 인프라 | 🟡 중 |
| 4 | templatefile | 인프라 | 🟡 중 |
| 5 | user_data 재실행 | 인프라 | 🟢 하 |
| 6 | NGC 이미지 없음 | Docker | 🔴 상 |
| <b>7</b> | <b>코어 패키지 누락</b> | <b>Docker</b> | <b>🔴 상</b> |
| 8 | Entrypoint 모드 | Docker | 🔴 상 |
| 9 | 스크립트 경로 | Docker | 🟡 중 |
| 10 | 빌드 격리 | Docker | 🔴 상 |
| 11 | Volume mount | Docker | 🟡 중 |
| 12 | 셰이더 캐시 | 런타임 | 🟢 하 |

---

👈 [Lab 7: 정리 및 다음 단계](07-cleanup.md)
👉 [Appendix B: 비용 분석 & 최적화](appendix-b-cost.md)
