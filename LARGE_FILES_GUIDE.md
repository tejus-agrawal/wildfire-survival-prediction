# Guide: Handling Large Files in Git

## Problem
Your repository contains **5.7GB of data files** and **401MB of model files**, which makes `git push` extremely slow. Git is not designed for large binary files.

## Solution Options

### ✅ Option 1: Don't Track Large Files (RECOMMENDED)
**Best for:** Most projects, especially academic/research projects

Large data files and model weights should NOT be in Git. Instead:

1. **Keep files locally** - They're already in your `.gitignore`
2. **Use cloud storage** for sharing:
   - Google Drive / Dropbox for one-time sharing
   - AWS S3 / Google Cloud Storage for production
   - Zenodo / Figshare for academic datasets
3. **Document in README** where to download the data

**Pros:**
- Fast Git operations
- No storage limits
- Standard practice for ML projects

**Cons:**
- Need separate storage solution
- Others need to download data separately

### Option 2: Git LFS (Limited Free Tier)
**Best for:** Small datasets (<1GB) or paid GitHub accounts

Git LFS stores large files separately but still tracks them in Git.

**GitHub LFS Limits:**
- Free: 1GB storage, 1GB bandwidth/month
- Pro: 50GB storage, 50GB bandwidth/month

**Your data:** 5.7GB exceeds free tier limits

**To use Git LFS (if you upgrade):**
```bash
# Track large file types
git lfs track "*.npy"
git lfs track "*.pth"
git lfs track "*.zip"

# Add the .gitattributes file
git add .gitattributes

# Add files normally (they'll use LFS)
git add data/mask_tensors/
git add models/
```

### Option 3: GitHub Releases
**Best for:** Versioned model checkpoints

Upload large files as GitHub Releases (up to 2GB per file, unlimited releases).

## Current Status

✅ **Completed:**
- Installed Git LFS
- Updated `.gitignore` to exclude large files
- Removed large files from Git tracking

## Next Steps

1. **Commit the changes:**
   ```bash
   git add .gitignore
   git commit -m "Remove large files from Git tracking"
   ```

2. **Push (will be much faster now!):**
   ```bash
   git push
   ```

3. **For sharing data:**
   - Add download instructions to README.md
   - Upload to cloud storage (Google Drive, etc.)
   - Or use GitHub Releases for model files

## Recommended README Addition

Add this to your README.md:

```markdown
## Data Download

The following large files are not included in this repository:

- `data/mask_tensors/` (20,940 .npy files, ~5.7GB)
- `data/images.zip` 
- `models/*.pth` (model weights, ~401MB)

**To download the data:**
1. [Add your download link here]
2. Extract to the `data/` directory
3. Model weights should go in `models/` directory
```

