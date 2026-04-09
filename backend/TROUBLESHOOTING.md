# Troubleshooting Guide for Railway Deployment

## Error: libgomp.so.1: cannot open shared object file

**Problem**: Missing system libraries required by scikit-learn and LightGBM.

**Solution**:
1. Ensure `apt.txt` exists in the backend folder with:
   ```
   libgomp1
   libgomp1-dev
   gcc
   g++
   ```

2. Force Railway to rebuild by:
   - Making a small change to any file
   - Or redeploying the project

3. Check Railway build logs to confirm apt packages are installed

## Error: Model loading fails

**Problem**: Model trained on different platform (Windows vs Linux).

**Solution**:
1. Retrain model on Linux environment if possible
2. Or ensure all required libraries are available in Railway

## Error: Import errors

**Problem**: Missing Python dependencies or incorrect paths.

**Solution**:
1. Check `requirements.txt` has all dependencies with versions
2. Verify file paths in code use relative paths from backend folder
3. Test locally: `python -c "from api.app import app"`

## Railway Deployment Checklist

- [ ] `backend/` folder contains all necessary files
- [ ] `Procfile` exists: `web: uvicorn api.app:app --host 0.0.0.0 --port $PORT`
- [ ] `requirements.txt` has pinned versions
- [ ] `apt.txt` exists for system dependencies
- [ ] `runtime.txt` specifies Python version
- [ ] Environment variables configured in Railway dashboard
- [ ] CORS allows Railway domain in backend