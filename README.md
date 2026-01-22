# Official Implementation of DNPR

## ⚠️ Repository Updating
We are currently in the process of uploading the complete source code and configuration files corresponding to the revised manuscript. 

To ensure the code is "out-of-the-box" executable on standard environments, we are finalizing the dependency checks and path configurations.

### 📢 Note to Reviewers
The repository is being updated in two stages:

#### 1. Immediate Updates (ETA: Within 48 Hours)
Our primary focus right now is the clean release of the **DNPR core framework**.
- **Core Logic:** The implementation of key modules (PMGR, NME, TPC) can currently be checked in the `lib/` directory.
- **Inference:** A standardized one-click inference script for the main results will be pushed shortly.

If you encounter any issues running the current version, please check back in 1-2 days.

#### 2. Roadmap & Future Releases (Ongoing)
To further support the community and ensure reproducibility of our comparative study, we are also organizing and planning to release the following components **in subsequent updates**:
- **Standardized Feature Extraction:** The custom ViT-L/14@336px extraction pipeline used in our experiments.
- **Baseline Reproduction:** Our reimplementations/configurations for the comparison methods mentioned in the paper:
    - [ ] WinCLIP & AnomalyCLIP
    - [ ] MuSc (Batch-Processing Version)
    - [ ] APRIL-GAN

*These components will be progressively uploaded to facilitate fair comparison benchmarks.*
