# Physical AI on AWS — Quadruped Locomotion RL Workshop

[English](README.md) | [한국어](README.ko.md)

> End-to-end reinforcement learning for ANYmal-C rough terrain locomotion using **Isaac Lab + PPO** on AWS.
> Provision GPU infrastructure with a single Terraform command and train 147M timesteps in just 75 minutes.

[![ANYmal-C Rough Terrain Locomotion](images/play30_frame_15s.png)](https://youtu.be/k98MgurW9y0)

*Trained ANYmal-C walking stably over rough block terrain (Play mode capture — click to watch video)*

---

## Highlights

| Item | Result |
|------|--------|
| **Robot** | ANYmal-C quadruped (12 joints) |
| **Environment** | Rough Terrain (rocks, slopes, stairs) |
| **Algorithm** | PPO (Proximal Policy Optimization) |
| **Parallel Envs** | 4,096 simultaneous simulations |
| **Training Time** | 75 min / 1,500 iterations |
| **Final Reward** | -0.50 → **+16.29** |
| **Episode Length** | 13 steps → **897 steps** (66x improvement) |
| **Terrain Level** | Reached 5.9 / 6.25 |
| **Total Cost** | **~$12** |
| **Outputs** | MP4 video + JIT/ONNX policy (sim-to-real ready) |

---

## Project Structure

```
pai-sim-isaaclab/
│
├── main.tf                    # Terraform: VPC, EC2, EBS, IAM, CloudWatch
├── variables.tf               # Input variable definitions
├── outputs.tf                 # Outputs (IP, SSH command)
├── terraform.tfvars.example   # Variable template (copy and fill in)
├── user_data.sh               # EC2 bootstrap script (Docker, Isaac Lab auto-install)
│
├── REPORT_Physical_AI_on_AWS.md   # Comprehensive lab report
├── isaac_lab_dashboard.html       # Training metrics dashboard (Chart.js)
│
├── images/                    # Screenshots & frame captures
│   ├── dashboard_screenshot*.png  # Training dashboard captures (3)
│   ├── play30_frame_*.png         # Play mode frames (7)
│   └── play_frame_*.png           # Initial play frames (5)
│
├── models/                    # Trained policy models
│   ├── policy_jit.pt              # TorchScript JIT (C++ inference)
│   └── policy.onnx                # ONNX (TensorRT/Jetson deployment)
│
├── videos/                    # Play mode recordings
│   ├── anymal_c_play.mp4          # 10s test video
│   └── anymal_c_play_30s.mp4      # 30s final video
│
└── workshop/                  # GitBook workshop (7 Labs + 3 Appendices)
    ├── README.md
    ├── SUMMARY.md
    ├── book.json
    ├── assets/                # Screenshots, frame captures
    └── chapters/
        ├── 01-concepts.md             # Physical AI core concepts
        ├── 02-infrastructure.md       # AWS GPU infrastructure setup
        ├── 03-docker-build.md         # Isaac Lab Docker build
        ├── 04-training.md             # PPO reinforcement learning
        ├── 05-results.md              # Training result analysis
        ├── 06-play-mode.md            # Play mode & policy export
        ├── 07-cleanup.md              # Cleanup & next steps
        ├── appendix-a-troubleshooting.md  # 12 real-world pitfalls
        ├── appendix-b-cost.md             # Cost analysis & optimization
        └── appendix-c-references.md       # SW versions, papers, glossary
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ AWS Cloud (ap-northeast-2 Seoul)                                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ VPC 10.0.0.0/16                                            │ │
│  │                                                            │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │ EC2: g6e.4xlarge                                     │  │ │
│  │  │                                                      │  │ │
│  │  │  GPU: NVIDIA L40S (48 GB VRAM)                      │  │ │
│  │  │  CPU: AMD EPYC 7R13 (16 vCPU, 128 GiB RAM)         │  │ │
│  │  │                                                      │  │ │
│  │  │  ┌────────────────────────────────────────────┐     │  │ │
│  │  │  │ Docker: isaac-lab-ready:latest (26.8 GB)   │     │  │ │
│  │  │  │  Isaac Sim 4.5.0 + Isaac Lab v2.1.0        │     │  │ │
│  │  │  │  RSL-RL (PPO) + PyTorch 2.5.1              │     │  │ │
│  │  │  └────────────────────────────────────────────┘     │  │ │
│  │  │                                                      │  │ │
│  │  │  /      : 300 GB gp3 (OS + Docker images)          │  │ │
│  │  │  /data  : 500 GB gp3 (Checkpoints + Datasets)      │  │ │
│  │  │  /scratch: NVMe instance store (Shader cache)       │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  │                          │                                 │ │
│  │                          ▼ (Auto-sync every 30 min)        │ │
│  │                 ┌─────────────────┐                        │ │
│  │                 │  S3 Checkpoint  │                        │ │
│  │                 └─────────────────┘                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  CloudWatch: GPU idle 30 min → Auto-stop EC2 (cost saving)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.5
- AWS CLI configured (`aws configure`)
- [NVIDIA NGC](https://ngc.nvidia.com/) API Key
- g6e instance service quota approved ([Service Quotas](https://console.aws.amazon.com/servicequotas/))

### 1. Deploy Infrastructure

```bash
# Create variable file
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — NGC API Key, SSH key, region, etc.

# Deploy (~3 min)
terraform init
terraform plan
terraform apply
```

### 2. SSH & Verify Bootstrap

```bash
ssh -i your-key.pem ubuntu@$(terraform output -raw public_ip)

# Monitor bootstrap progress (15-25 min)
tail -f /var/log/isaac-lab-setup.log
# Wait for "Isaac Lab setup COMPLETE" message
```

### 3. Complete Isaac Lab Docker Image

`user_data.sh` automatically pulls Isaac Sim and builds Isaac Lab.
However, the **core `isaaclab` package requires manual installation**:

```bash
docker run --name setup --gpus all \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  --entrypoint bash isaac-lab-base:latest -c '
    /workspace/isaaclab/_isaac_sim/python.sh -m pip install \
      --no-build-isolation -e /workspace/isaaclab/source/isaaclab
    cd /workspace/isaaclab && ./isaaclab.sh -i rsl_rl
  '
docker commit setup isaac-lab-ready:latest
docker rm setup
```

> **Why?** `docker compose --profile base build` only installs extension packages and misses the core `isaaclab` package. Without this step, you'll get `ModuleNotFoundError: No module named 'isaaclab'`.

### 4. Run Training

```bash
docker run --rm --gpus all --network=host \
  --entrypoint /workspace/isaaclab/isaaclab.sh \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  -v "/scratch/isaac-sim-cache/kit:/isaac-sim/kit/cache:rw" \
  -v "/data/checkpoints:/workspace/isaaclab/logs:rw" \
  isaac-lab-ready:latest \
  -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless
```

> **Important**: The `--entrypoint` override is required! The default entrypoint (`runheadless.sh`) starts a streaming server.

### 5. Play Mode (Visualize Trained Policy)

```bash
docker run --rm --gpus all --network=host \
  --entrypoint /workspace/isaaclab/isaaclab.sh \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  -v "/scratch/isaac-sim-cache/kit:/isaac-sim/kit/cache:rw" \
  -v "/data/checkpoints:/workspace/isaaclab/logs:rw" \
  isaac-lab-ready:latest \
  -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --headless --video --video_length 1500 --num_envs 16 \
    --load_run <TIMESTAMP_DIR>
```

Outputs:
- `rl-video-step-0.mp4` — Locomotion video (30s, 1280x720)
- `exported/policy.pt` — TorchScript JIT (C++ real-time inference)
- `exported/policy.onnx` — ONNX (TensorRT/Jetson deployment)

### 6. Cleanup

```bash
terraform destroy   # Delete all AWS resources
```

---

## Training Results

### Learning Curve

| Phase | Iterations | Mean Reward | Description |
|-------|-----------|-------------|-------------|
| Exploration | 0-40 | -0.5 → -4.9 | Random exploration, penalties activate |
| Foundation | 40-120 | -4.9 → +5.0 | Gait pattern acquisition, reward crosses 0 |
| Refinement | 120-300 | +5.0 → +15.0 | Stable walking, velocity tracking improves |
| Convergence | 300-1500 | +15.0 → +16.3 | Policy converges, terrain difficulty increases |

### Dashboard

![Training Dashboard - Reward & Episode Length](images/dashboard_screenshot.png)

![Training Dashboard - Reward Components](images/dashboard_screenshot2.png)

![Training Dashboard - Policy & Loss](images/dashboard_screenshot3.png)

### Play Mode

[![Play Mode - 5s](images/play30_frame_5s.png)](https://youtu.be/k98MgurW9y0)
[![Play Mode - 15s](images/play30_frame_15s.png)](https://youtu.be/k98MgurW9y0)
[![Play Mode - 25s](images/play30_frame_25s.png)](https://youtu.be/k98MgurW9y0)

---

## Real-World Troubleshooting (12 Pitfalls)

Hard-won lessons from actual deployment — not found in official docs:

| # | Pitfall | Severity | Key Fix |
|---|---------|----------|---------|
| 1 | dpkg lock contention | :yellow_circle: | `systemctl stop unattended-upgrades` + `DPkg::Lock::Timeout=120` |
| 2 | EBS device name (Nitro) | :yellow_circle: | Dynamic `/dev/nvme*n1` discovery (not xvdf) |
| 3 | Instance store already mounted | :yellow_circle: | Reuse `/opt/dlami/nvme` |
| 4 | Terraform templatefile conflict | :yellow_circle: | `$${VAR}` double dollar escape |
| 5 | user_data won't re-run | :green_circle: | `terraform taint` → recreate instance |
| 6 | No Isaac Lab NGC image | :red_circle: | Build from source: `docker compose --profile base build` |
| **7** | **Core isaaclab package missing** | **:red_circle:** | **`pip install --no-build-isolation -e source/isaaclab`** |
| 8 | Docker entrypoint streaming mode | :red_circle: | `--entrypoint /workspace/isaaclab/isaaclab.sh` |
| 9 | Training script path changed (v2.1.0) | :yellow_circle: | `scripts/reinforcement_learning/rsl_rl/train.py` |
| 10 | setuptools build isolation | :red_circle: | `--no-build-isolation` flag |
| 11 | Volume mount breaks editable install | :yellow_circle: | Don't mount source directories |
| 12 | Shader cache 4-min first-run delay | :green_circle: | Mount cache volume + patience |

> Details in [workshop/chapters/appendix-a-troubleshooting.md](workshop/chapters/appendix-a-troubleshooting.md)

---

## Cost

| Scenario | Instance | Region | Est. Cost |
|----------|----------|--------|-----------|
| **This workshop** | g6e.4xlarge On-Demand | Seoul | **~$12** |
| Cost optimized | g6e.4xlarge Spot | Virginia | ~$2.50 |
| Large scale | g6e.12xlarge Spot | Virginia | ~$8 |

**Cost saving features:**
- GPU idle 30 min → CloudWatch auto-stop
- S3 checkpoint auto-sync (Spot interruption protection)

---

## Workshop

A step-by-step GitBook workshop is included in the `workshop/` directory:

```bash
cd workshop
npm install honkit
npx honkit serve
# Open http://localhost:4000
```

| Lab | Topic | Duration |
|-----|-------|----------|
| Lab 1 | Physical AI Core Concepts | 10 min |
| Lab 2 | Terraform AWS GPU Infrastructure | 20 min |
| Lab 3 | Isaac Lab Docker Image Build | 30 min |
| Lab 4 | PPO Reinforcement Learning Training | 75 min |
| Lab 5 | Training Result Analysis | 15 min |
| Lab 6 | Play Mode & Policy Export | 10 min |
| Lab 7 | Cleanup & Next Steps | 10 min |
| Appendix A | 12 Real-World Pitfalls | — |
| Appendix B | Cost Analysis & Optimization | — |
| Appendix C | SW Versions, Papers, Glossary | — |

---

## Software Versions

| Software | Version |
|----------|---------|
| Ubuntu | 22.04 LTS |
| NVIDIA Driver | 580.126.09 |
| Isaac Sim | 4.5.0 |
| Isaac Lab | v2.1.0 |
| PyTorch | 2.5.1 |
| RSL-RL | 2.x |
| Terraform | >= 1.5 |

---

## License

The Terraform code and workshop documentation in this project are released under the MIT License.
Isaac Sim / Isaac Lab are subject to the [NVIDIA EULA](https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html).

---

> **This project is based on real AWS deployment experience in April 2026.**
> Completed end-to-end quadruped locomotion reinforcement learning for approximately $12.
