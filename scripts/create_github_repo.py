#!/usr/bin/env python3
"""
Script to create a GitHub repository for CasaLingua.

This script uses the GitHub API to create a new repository with the specified
settings and adds the local repository as a remote.

Usage:
    python create_github_repo.py [--name REPO_NAME] [--description DESC] [--private]

Requirements:
    pip install requests
"""

import argparse
import json
import os
import subprocess
import sys

import requests


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a GitHub repository for CasaLingua")
    parser.add_argument(
        "--name", 
        type=str, 
        default="casalingua", 
        help="Name of the repository (default: casalingua)"
    )
    parser.add_argument(
        "--description", 
        type=str, 
        default="CasaLingua - Multilingual Translation and Simplification Platform", 
        help="Repository description"
    )
    parser.add_argument(
        "--private", 
        action="store_true", 
        help="Make the repository private (default: public)"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        help="GitHub Personal Access Token (or set GITHUB_TOKEN env variable)"
    )
    parser.add_argument(
        "--organization", 
        type=str, 
        help="GitHub organization name (if creating under an organization)"
    )
    
    return parser.parse_args()


def create_repository(name, description, private, token, organization=None):
    """Create a GitHub repository."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "name": name,
        "description": description,
        "private": private,
        "has_issues": True,
        "has_projects": True,
        "has_wiki": True,
        "auto_init": False,
        "allow_squash_merge": True,
        "allow_merge_commit": True,
        "allow_rebase_merge": True,
        "delete_branch_on_merge": True
    }
    
    # Create in organization or user account
    if organization:
        url = f"https://api.github.com/orgs/{organization}/repos"
    else:
        url = "https://api.github.com/user/repos"
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 201:
        repo_data = response.json()
        print(f"Repository created successfully: {repo_data['html_url']}")
        return repo_data
    else:
        print(f"Error creating repository: {response.status_code}")
        print(response.json())
        sys.exit(1)


def update_git_remote(repo_url, remote_name="origin"):
    """Update the Git remote URL."""
    # Check if remote exists
    result = subprocess.run(
        ["git", "remote", "-v"], 
        capture_output=True, 
        text=True
    )
    
    if remote_name in result.stdout:
        print(f"Remote {remote_name} already exists. Updating URL...")
        subprocess.run(["git", "remote", "set-url", remote_name, repo_url])
    else:
        print(f"Adding remote {remote_name}...")
        subprocess.run(["git", "remote", "add", remote_name, repo_url])
    
    print(f"Remote {remote_name} set to {repo_url}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Get token from args or environment
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GitHub token is required.")
        print("Either provide --token argument or set GITHUB_TOKEN environment variable.")
        sys.exit(1)
    
    # Create the repository
    repo_data = create_repository(
        args.name, 
        args.description, 
        args.private, 
        token,
        args.organization
    )
    
    # Update Git remote
    update_git_remote(repo_data["clone_url"])
    
    print("\nNext steps:")
    print("1. Push your local repository:")
    print("   git push -u origin main")
    print("2. Set up branch protection rules on GitHub")
    print("3. Enable GitHub Actions in repository settings")
    print(f"4. Access your repository at: {repo_data['html_url']}")


if __name__ == "__main__":
    main()