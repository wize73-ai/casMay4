#!/bin/bash
# Script to push CasaLingua to GitHub

# Colors
RED=$(tput setaf 1 2>/dev/null || echo '')
GREEN=$(tput setaf 2 2>/dev/null || echo '')
BLUE=$(tput setaf 4 2>/dev/null || echo '')
CYAN=$(tput setaf 6 2>/dev/null || echo '')
YELLOW=$(tput setaf 3 2>/dev/null || echo '')
BOLD=$(tput bold 2>/dev/null || echo '')
NC=$(tput sgr0 2>/dev/null || echo '') # Reset

# Box drawing chars
TL="╔"
TR="╗"
BL="╚"
BR="╝"
HORIZ="═"
VERT="║"

# Print banner
echo -e "${CYAN}"
echo -e "${TL}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${TR}"
echo -e "${VERT}                                                                              ${VERT}"
echo -e "${VERT}   ${CYAN}   _____                _      _                          ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN}  / ____|              | |    (_)                         ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN} | |     __ _ ___  __ _| |     _ _ __   __ _ _   _  __ _  ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN} | |    / _\` / __|/ _\` | |    | | '_ \\ / _\` | | | |/ _\` | ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN} | |___| (_| \\__ \\ (_| | |____| | | | | (_| | |_| | (_| | ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN}  \\_____\\__,_|___/\\__,_|______|_|_| |_|\\__, |\\__,_|\\__,_| ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN}                                         __/ |             ${NC}                                   ${VERT}"
echo -e "${VERT}   ${CYAN}                                        |___/              ${NC}                                   ${VERT}"
echo -e "${VERT}                                                                              ${VERT}"
echo -e "${VERT}                 ${YELLOW}GitHub Repository Setup${NC}                                       ${VERT}"
echo -e "${VERT}                                                                              ${VERT}"
echo -e "${BL}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${HORIZ}${BR}"
echo ""

# Check remote
echo -e "${BLUE}Checking for existing remote...${NC}"
git remote -v

echo ""
echo -e "${BLUE}Would you like to:${NC}"
echo "1. Push to existing remote (wize73-ai/casMay4)"
echo "2. Create a new GitHub repository (requires GitHub token)"
echo "3. Cancel"
read -p "Select an option (1-3): " choice

case $choice in
    1)
        echo -e "${BLUE}Pushing to existing remote...${NC}"
        git push -u origin main
        echo -e "${GREEN}✅ Push complete!${NC}"
        ;;
    2)
        # Check if GitHub API token is set
        if [ -z "${GITHUB_TOKEN}" ]; then
            echo -e "${YELLOW}GitHub token not found in environment.${NC}"
            read -p "Enter your GitHub Personal Access Token: " token
            export GITHUB_TOKEN="$token"
        fi
        
        echo -e "${BLUE}Setting up a new GitHub repository...${NC}"
        read -p "Repository name (default: casalingua): " repo_name
        repo_name=${repo_name:-casalingua}
        
        read -p "Repository description (default: CasaLingua - Multilingual Translation Platform): " repo_desc
        repo_desc=${repo_desc:-"CasaLingua - Multilingual Translation Platform"}
        
        read -p "Make repository private? (y/N): " is_private
        if [[ $is_private == "y" || $is_private == "Y" ]]; then
            private_flag="--private"
        else
            private_flag=""
        fi
        
        read -p "Create under organization? (leave blank for personal account): " org_name
        if [ -n "$org_name" ]; then
            org_flag="--organization $org_name"
        else
            org_flag=""
        fi
        
        echo -e "${BLUE}Creating repository...${NC}"
        python scripts/create_github_repo.py --name "$repo_name" --description "$repo_desc" $private_flag $org_flag
        
        if [ $? -eq 0 ]; then
            echo -e "${BLUE}Pushing code to the new repository...${NC}"
            git push -u origin main
            echo -e "${GREEN}✅ Repository created and code pushed successfully!${NC}"
        else
            echo -e "${RED}Error creating GitHub repository.${NC}"
            exit 1
        fi
        ;;
    3)
        echo -e "${YELLOW}Operation cancelled.${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid option.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}GitHub Repository Setup Complete!${NC}"
echo -e "Don't forget to set up these items on GitHub:"
echo -e "• Branch protection rules"
echo -e "• GitHub Actions permissions"
echo -e "• Secrets for GitHub Actions (if needed)"
echo ""