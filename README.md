# Simple AI Agent

A minimal project with a FastAPI backend and a React frontend.

## Structure

- `agent/`: FastAPI backend
- `frontend/`: React frontend

## Usage

Build and run with Docker Compose:

```sh
docker-compose up --build
```

## CI/CD

This project uses GitHub Actions for CI/CD. There are two workflows:

- **Backend**: Builds and pushes the backend Docker image from `agent/`.
- **Frontend**: Builds and pushes the frontend Docker image from `frontend/`.

See `.github/workflows/` for workflow details.
