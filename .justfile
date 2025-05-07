# Like GNU `make`, but `just` rustier.
# https://just.systems/
# run `just` from this directory to see available commands

alias i := install
alias p := pre_commit
alias r := run
alias c := clean
alias ch := check
alias a := add_scripts

# Default command when 'just' is run without arguments
default:
  @just --list

# Install the virtual environment and pre-commit hooks
install:
  uv sync
  uv run pre-commit install

# Run pre-commit
pre_commit:
  uv run pre-commit run -a

# Run a package with specified architecture
# Usage: just run [arch=x86_64]
run *args='core':
  uv run {{args}}

# Clean the project
clean:
  @# Remove cached files
  @find . -type d -name "__pycache__" -exec rm -r {} +
  @find . -type d -name "*.egg-info" -exec rm -r {} +

# Run code quality tools
check:
  @# Check lock file consistency
  uv lock --locked
  @# Run pre-commit
  uv run pre-commit run -a
  @# Run mypy
  uv run mypy .
  @# Run deptry with ignored issues
  uv run deptry . --ignore=DEP002,DEP003

# Add scripts
add_scripts:
  uv add --script scripts/this.py 'typer>=0.12.5'
