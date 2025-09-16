# Git Workflow Guide - Traffic Prediction Demo

## Repository Setup

### Initial Setup (Already Done)
```bash
git init
git add .
git commit -m "Initial commit: Traffic Prediction Demo"
```

## Development Workflow

### 1. Feature Development
```bash
# Create new feature branch
git checkout -b feature/new-feature-name

# Make changes
# Edit files...

# Stage changes
git add .

# Commit changes
git commit -m "Add: description of new feature"

# Switch back to main branch
git checkout master

# Merge feature branch
git merge feature/new-feature-name

# Delete feature branch (optional)
git branch -d feature/new-feature-name
```

### 2. Bug Fixes
```bash
# Create bug fix branch
git checkout -b fix/bug-description

# Fix the bug
# Edit files...

# Commit fix
git commit -m "Fix: description of bug fix"

# Merge back to main
git checkout master
git merge fix/bug-description
```

### 3. Documentation Updates
```bash
git checkout -b docs/update-description

# Update documentation
# Edit README.md, USAGE.md, etc.

git add .
git commit -m "Docs: update documentation"

git checkout master
git merge docs/update-description
```

## Commit Message Convention

### Format
```
Type: Short description

Optional longer description explaining what and why
```

### Types
- **Add**: New features or files
- **Fix**: Bug fixes
- **Update**: Improvements to existing features
- **Docs**: Documentation changes
- **Refactor**: Code restructuring without functionality change
- **Style**: Formatting, missing semicolons, etc.
- **Test**: Adding or updating tests
- **Chore**: Maintenance tasks

### Examples
```bash
git commit -m "Add: sample data selection feature in prediction page"
git commit -m "Fix: timestamp parsing error in create_sample_data_from_timestamp"
git commit -m "Update: improve prediction accuracy visualization"
git commit -m "Docs: add troubleshooting guide for common errors"
```

## Useful Git Commands

### Status and Information
```bash
git status                 # Check working directory status
git log --oneline         # View commit history
git diff                  # See changes in working directory
git diff --staged         # See staged changes
git show HEAD             # Show last commit details
```

### Branching
```bash
git branch                # List all branches
git branch <name>         # Create new branch
git checkout <name>       # Switch to branch
git checkout -b <name>    # Create and switch to new branch
git branch -d <name>      # Delete branch
```

### Undoing Changes
```bash
git checkout -- <file>    # Discard changes in working directory
git reset HEAD <file>     # Unstage file
git reset --soft HEAD~1   # Undo last commit, keep changes staged
git reset --hard HEAD~1   # Undo last commit, discard changes
```

### Remote Repository (for future use)
```bash
git remote add origin <url>           # Add remote repository
git push -u origin master             # Push to remote for first time
git push                              # Push changes to remote
git pull                              # Pull changes from remote
git clone <url>                       # Clone repository
```

## Branch Strategy

### Main Branches
- **master**: Production-ready code
- **develop**: Integration branch for features (optional)

### Supporting Branches
- **feature/**: New features
- **fix/**: Bug fixes
- **hotfix/**: Urgent fixes for production
- **docs/**: Documentation updates

## File Management

### What to Track
- ✅ Source code (app.py)
- ✅ Documentation (README.md, USAGE.md, etc.)
- ✅ Configuration (requirements.txt)
- ✅ Model files (xgb_model.pkl) - if not too large
- ✅ Sample data (traffic_dataset1.csv) - if reasonable size

### What NOT to Track (in .gitignore)
- ❌ Virtual environments (.venv/)
- ❌ Python cache (__pycache__/)
- ❌ IDE files (.vscode/, .idea/)
- ❌ OS files (.DS_Store)
- ❌ Logs (*.log)
- ❌ Temporary files (*.tmp)

## Backup and Recovery

### Create Tags for Releases
```bash
git tag -a v1.0 -m "Version 1.0: Initial release"
git tag -a v1.1 -m "Version 1.1: Bug fixes and improvements"
git show v1.0              # Show tag details
```

### Backup Repository
```bash
# Create archive
git archive --format=zip --output=traffic-demo-backup.zip HEAD

# Or create bare clone
git clone --bare . ../traffic-demo-backup.git
```

## Collaboration

### For Team Development
1. Clone repository: `git clone <repository-url>`
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and commit
4. Push branch: `git push origin feature/your-feature`
5. Create pull request (on GitHub/GitLab)
6. Review and merge

### Sync with Remote
```bash
git fetch origin          # Get latest changes without merging
git pull origin master    # Fetch and merge latest changes
git push origin master    # Push your changes to remote
```

## Troubleshooting

### Common Issues
1. **Merge Conflicts**: Edit files manually, then `git add` and `git commit`
2. **Accidentally Committed**: Use `git reset` to undo
3. **Wrong Branch**: Use `git checkout` to switch
4. **Lost Changes**: Check `git reflog` for recovery

### Emergency Commands
```bash
git reflog                 # See all commits including deleted ones
git reset --hard <commit>  # Reset to specific commit
git clean -fd              # Remove untracked files and directories
```
