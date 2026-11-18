# Deploy Fake News System on Render (SQLite)

This project is prepared for deployment on Render using **SQLite only**.

## Important Notes about SQLite on Render Free Tier

- The disk is **ephemeral**. Every time the service restarts or is redeployed, the SQLite database and any newly uploaded files are lost.
- Trained models that are already in `media/models/` will survive as long as they are committed in the Git repository.
- For production use with persistent data, consider upgrading to a paid plan with a Persistent Disk, or switch to PostgreSQL later.

## Python Version

This project pins the Render runtime to **Python 3.11.11** using the `.python-version` file. Keep this file in the repository root. Do not override it with a conflicting `PYTHON_VERSION` environment variable in Render.

## Quick Deploy Steps

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Ready for Render deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Create Web Service on Render

1. Go to https://dashboard.render.com → **New +** → **Web Service**
2. Connect your GitHub repository
3. Configure:

| Setting            | Value                                              |
|--------------------|----------------------------------------------------|
| **Name**           | `fake-news-system` (any name)                      |
| **Runtime**        | Python 3                                           |
| **Build Command**  | `chmod +x build.sh && ./build.sh`                  |
| **Start Command**  | `gunicorn news_analysis.wsgi:application`          |
| **Instance Type**  | Free                                               |

### 3. Environment Variables (Required)

Add these in the Render dashboard under **Environment**:

| Key                    | Value                                      |
|------------------------|--------------------------------------------|
| `DJANGO_SECRET_KEY`    | A long random string (generate one)        |
| `DJANGO_DEBUG`         | `0`                                        |
| `DJANGO_ALLOWED_HOSTS` | `your-service-name.onrender.com`           |

Generate a secret key:

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 4. Deploy

Click **Create Web Service**. Wait for the build to finish.

Your app will be available at:
`https://your-service-name.onrender.com`

## Local Testing of Production Settings

```bash
export DJANGO_DEBUG=0
export DJANGO_SECRET_KEY=some-secret-key
export DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
python manage.py collectstatic
gunicorn news_analysis.wsgi:application
```

## What is Included

- Trained models in `media/models/` (so predictions work immediately)
- Small sample CSV files only
- WhiteNoise for static files
- Gunicorn as the production server
- Automatic NLTK data download during build
- Migrations run automatically during build
