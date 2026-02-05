# Quick Reference Card

## 🚀 Quick Start Commands

### Full Pipeline (Docker Compose)
```bash
docker-compose up                    # Run all phases
docker-compose up analysis           # Phase 1 only
docker-compose up training           # Phase 2 only
docker-compose up evaluation         # Phase 3 only
docker-compose up dashboard          # Interactive dashboard
```

### Individual Containers

#### Phase 1: Data Analysis
```bash
cd data_analysis
docker build -t bdd-analysis .
docker run -v $(pwd)/../bdd100k_images_100k:/data/images:ro \
           -v $(pwd)/../bdd100k_labels:/data/labels:ro \
           -v $(pwd)/../results/analysis:/app/results \
           bdd-analysis
```

#### Phase 2: Training (GPU)
```bash
cd model
docker build -t bdd-training .
docker run --gpus all \
           -v $(pwd)/../bdd100k_images_100k:/data/images:ro \
           -v $(pwd)/../bdd100k_labels:/data/labels:ro \
           -v $(pwd)/../results/checkpoints:/checkpoints \
           bdd-training --epochs 10 --batch-size 4
```

#### Phase 3: Evaluation
```bash
cd evaluation
docker build -t bdd-evaluation .
docker run -v $(pwd)/../bdd100k_images_100k:/data/images:ro \
           -v $(pwd)/../bdd100k_labels:/data/labels:ro \
           -v $(pwd)/../results/checkpoints:/checkpoints:ro \
           -v $(pwd)/../results/metrics:/metrics \
           bdd-evaluation --checkpoint /checkpoints/best_model.pth
```

## 📁 Project Structure
```
BDD_Detection_Task/
├── docker-compose.yml              # Multi-container orchestration
├── DOCKER_GUIDE.md                 # Comprehensive Docker guide
├── data_analysis/
│   ├── Dockerfile                  # Phase 1 container
│   ├── requirements.txt
│   ├── README.md                   # Module documentation
│   └── *.py                        # Analysis scripts
├── model/
│   ├── Dockerfile                  # Phase 2 container
│   ├── requirements.txt
│   ├── README.md                   # Module documentation
│   └── *.py                        # Training scripts
├── evaluation/
│   ├── Dockerfile                  # Phase 3 container
│   ├── requirements.txt
│   ├── README.md                   # Module documentation
│   └── *.py                        # Evaluation scripts
└── results/
    ├── analysis/                   # Phase 1 outputs
    ├── checkpoints/                # Phase 2 outputs
    ├── logs/                       # Training logs
    ├── metrics/                    # Phase 3 metrics
    └── visualizations/             # Phase 3 visualizations
```

## 🔧 Common Commands

### Build & Run
```bash
docker-compose build                # Build all containers
docker-compose build training       # Build specific service
docker-compose up -d                # Run in background
docker-compose down                 # Stop all containers
docker-compose down -v              # Stop and remove volumes
```

### Monitoring
```bash
docker ps                           # List running containers
docker-compose logs -f training     # Follow training logs
docker stats                        # Container resource usage
docker exec -it <container> bash    # Open shell in container
```

### Cleanup
```bash
docker system prune -a              # Remove all unused images
docker volume prune                 # Remove unused volumes
docker-compose down -v              # Remove all project volumes
```

## 📊 Expected Outputs

### Phase 1 (Analysis)
- `results/analysis/class_distribution.json`
- `results/analysis/anomalies.json`
- Dashboard at http://localhost:8501

### Phase 2 (Training)
- `results/checkpoints/best_model.pth`
- `results/checkpoints/checkpoint_epoch_N.pth`
- `results/logs/training.log`

### Phase 3 (Evaluation)
- `results/metrics/evaluation_results.json`
- `results/visualizations/viz_*.jpg`
- `results/visualizations/confusion_matrix.png`

## 🎯 GPU Configuration

### Enable GPU in docker-compose.yml
```yaml
training:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Test GPU Access
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 📖 Documentation Links

- [Main README](README.md)
- [Docker Guide](DOCKER_GUIDE.md)
- [Data Analysis Module](data_analysis/README.md)
- [Model Training Module](model/README.md)
- [Evaluation Module](evaluation/README.md)

## 💡 Tips

1. **Use Docker Compose** for full pipeline automation
2. **Enable GPU** for 10-20x faster training
3. **Start small** with `--subset-size 100 --epochs 1` for testing
4. **Monitor logs** with `docker-compose logs -f <service>`
5. **Clean up** regularly with `docker system prune -a`

---

**Quick Help**: For detailed instructions, see individual module README files or [DOCKER_GUIDE.md](DOCKER_GUIDE.md)
