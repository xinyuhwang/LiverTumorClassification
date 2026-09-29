# Running HERALD on AICR

Setup guide for the Massachusetts AI Compute Resource. It is based on [docs.aicr.ai](https://docs.aicr.ai/), checked 2026-09-29. If anything here disagrees with the official docs, trust the docs.

## Is the data allowed on AICR?

Yes. AICR's acceptable-use policy forbids non-anonymized data, and data covered by a data use agreement. MCT-LTDiag ([doi:10.7910/DVN/S3RW15](https://doi.org/10.7910/DVN/S3RW15)) is:
- de-identified
- released under **CC0 1.0**
- free of restricted files and use agreements

Check again if you ever add other data, for example hospital data under a data use agreement.

## Cluster facts

| | |
|---|---|
| Login | `login.aicr.ai`. SSH **certificate** from [ood.aicr.ai](https://ood.aicr.ai); no password, no VPN. |
| Username | `<institution username>_<institution code>` |
| Data transfer | `dtn0001.aicr.ai`, `dtn0002.aicr.ai` (rsync/scp), or Globus. The login node caps users at 4 cores, so don't transfer through it. |
| GPUs | `rtx-batch` (RTX PRO 6000, ≤24 h), `b200-batch` (B200, ≤24 h), `rtx-devel` / `b200-devel` (≤4 h, interactive), `preemptable` |
| CPU | `cpu` partition (≤24 h) |
| Storage | `/home/$USER`: 100 GiB, snapshotted. `/scratch/$USER`: 10 TiB, **files older than 30 days are purged**. `/work/<inst>/<group>`: group space, snapshotted. Nothing is backed up outside AICR, and storage is meant to hold active projects only. |
| Rules | No computation on login nodes. Use `sbatch`, or `salloc` for interactive work. |

### Where the project keeps things

| What | Where (set in `cluster/env.sh`) | Why |
|---|---|---|
| Code | `$HOME/LiverTumorClassification` | Small, snapshotted |
| Python env | `$HOME/envs/herald` | Reused by every job |
| Raw archives (180 GB) | `$HERALD_STORE/raw` on `/work` | Slow to re-download; not purged |
| Prepared data (~30 GB) | `$HERALD_STORE/mct_ltdiag` on `/work` | Input to every run |
| Caches, checkpoints | `$HERALD_SCRATCH/work` on `/scratch` | Large, rebuildable |
| Finished runs | `$HERALD_STORE/results` on `/work` | Copied from scratch by each job |

If your group has no `/work` space yet, set `HERALD_STORE` to `/scratch/$USER/herald_store`. Scratch purges files after 30 days, so pull results out regularly.

## One-time setup

**1. Get an account.** Request access through [docs.aicr.ai/request-access](https://docs.aicr.ai/request-access/) (Northeastern is a member institution). You'll also need the Slurm account name of your allocation.

**2. SSH certificate (on your laptop).** Log in to [ood.aicr.ai](https://ood.aicr.ai) and download the key files. Put `id_ed25519_aicr`, `id_ed25519_aicr.pub` and `id_ed25519_aicr-cert.pub` in `~/.ssh/`, then:

```bash
chmod 600 ~/.ssh/id_ed25519_aicr
chmod 644 ~/.ssh/id_ed25519_aicr.pub ~/.ssh/id_ed25519_aicr-cert.pub
ssh-keygen -p -f ~/.ssh/id_ed25519_aicr        # set your own passphrase
cat cluster/ssh_config.example >> ~/.ssh/config   # then replace AICR_USERNAME
ssh aicr                                          # test
```

Certificates are short-lived. When `ssh` stops accepting the key, download a new certificate from OnDemand.

**3. Code (on AICR).**

```bash
ssh aicr
git clone <this repository's URL> ~/LiverTumorClassification
cd ~/LiverTumorClassification
sacctmgr show user $USER withassoc format=user,account -p   # find your account
nano cluster/env.sh      # set HERALD_ACCOUNT and HERALD_STORE
```

**4. Python environment (inside a GPU session).**

```bash
salloc --partition=rtx-devel --gpus=1 --cpus-per-task=4 --mem=16G --time=00:45:00
bash cluster/setup_env.sh      # ends by printing the GPU name and "matmul ok: True"
exit
```

**5. Download the dataset on the data-transfer node.** It's about 180 GB and resumes if interrupted. Run it inside `tmux` so it survives disconnects:

```bash
ssh aicr-dtn
tmux new -s download
source ~/LiverTumorClassification/cluster/env.sh
python3 ~/LiverTumorClassification/data_prep/download_mct_ltdiag.py --out "$HERALD_RAW" --workers 4
# detach with Ctrl-b d; reattach later with: tmux attach -t download
```

If the DTN has no `tmux` or `python3`, use Globus instead. Globus can't pull from Dataverse directly, so the download script remains the simplest route.

**6. Prepare the data (CPU job).**

```bash
bash cluster/submit.sh prepare_data
# check $HERALD_DATA/manifest.csv — the job log lists any flagged cases
```

## Everyday use

```bash
ssh aicr && cd ~/LiverTumorClassification && git pull
bash cluster/submit.sh ds2net --stage 1 --run_name e01_v1_liver
squeue --me                                     # queue status
tail -f /scratch/$USER/herald/logs/herald-ds2net-<jobid>.out
bash cluster/submit.sh ds2net --stage 1 --run_name e01_v1_liver --resume   # after a 24 h timeout
```

To try a job on the 4-hour devel partition first:

```bash
SBATCH_EXTRA="--partition=rtx-devel --time=01:00:00" \
    bash cluster/submit.sh ds2net --stage 1 --run_name smoke --smoke_test
```

Bring results back to your laptop, into the experiment folder:

```bash
bash cluster/pull_results.sh $HERALD_STORE/results/runs/ds2net/e01_v1_liver \
     experiments/E01_liver_labels/v1_official_liver_masks/results
```

## Files

| File | Purpose |
|---|---|
| `cluster/env.sh` | Account, storage paths, modules, venv. Sourced by everything. |
| `cluster/setup_env.sh` | Creates the Python environment and checks the GPU |
| `cluster/ssh_config.example` | `aicr` and `aicr-dtn` SSH host entries |
| `cluster/submit.sh` | `sbatch` wrapper that adds your account and log path |
| `cluster/jobs/*.sbatch` | `prepare_data`, `ds2net`, `unet_hybrid`, `stage3_paper` |
| `cluster/pull_results.sh` | Laptop-side copy of run metrics into `experiments/` |
