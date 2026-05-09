---
name: github-workflow
version: 1.0.0
description: GitHub operations: clone repos, manage PRs, create issues, configure CI/CD, and handle git workflows.
---
# GitHub Workflow Skill

## When to use

When the user asks about GitHub operations: cloning repos, creating PRs, managing
issues, configuring CI/CD, or performing git workflows.

## Common Operations

### Clone a repository
```bash
git clone https://github.com/<user>/<repo>.git ~/workspace/<repo>
```

### Create a branch and push
```bash
git checkout -b feature/<name>
git add <files>
git commit -m "type: description"
git push origin feature/<name>
```

### Create a PR via GitHub API
```bash
curl -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/<user>/<repo>/pulls \
  -d '{"title":"...", "head":"feature/...", "base":"master", "body":"..."}'
```

### Commit message conventions
- `feat:` — New feature
- `fix:` — Bug fix
- `refactor:` — Code restructuring
- `test:` — Adding/updating tests
- `docs:` — Documentation changes
- `chore:` — Maintenance tasks

## Pitfalls

- Always pull before pushing to avoid conflicts
- Use `--force-with-lease` instead of `--force` for safety
- GitHub API rate limit: 5000/hr authenticated, 60/hr unauthenticated
- Use token from `~/.hermes/.env` or `GITHUB_TOKEN` env var, never hardcode
