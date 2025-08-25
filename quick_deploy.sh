#!/bin/bash

# 🚀 factr.ai Quick Deployment Script
# This script automates the deployment process for factr.ai

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo -e "${BLUE}"
cat << "EOF"
    ____           __                  _ 
   / __/___ ______/ /______  ____ _   (_)
  / /_/ __ `/ ___/ __/ ___/ / __ `/  / / 
 / __/ /_/ / /__/ /_/ /  _ / /_/ /  / /  
/_/  \__,_/\___/\__/_/  (_)\__,_/  /_/   

Advanced Multimodal Misinformation Detection
Session 4 - Production Ready Deployment
EOF
echo -e "${NC}"

print_status "Starting factr.ai deployment process..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Docker
install_docker() {
    print_status "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    print_success "Docker installed successfully"
}

# Function to install Docker Compose
install_docker_compose() {
    print_status "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    print_success "Docker Compose installed successfully"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check for git
    if ! command_exists git; then
        print_status "Installing git..."
        sudo apt update && sudo apt install -y git
    fi
    
    # Check for curl
    if ! command_exists curl; then
        print_status "Installing curl..."
        sudo apt update && sudo apt install -y curl
    fi
    
    # Check for Docker
    if ! command_exists docker; then
        install_docker
        print_warning "Docker installed. Please log out and log back in, then run this script again."
        exit 0
    fi
    
    # Check for Docker Compose
    if ! command_exists docker-compose; then
        install_docker_compose
    fi
    
    print_success "All prerequisites are satisfied"
}

# Function to get deployment type
get_deployment_type() {
    echo ""
    echo "Choose deployment type:"
    echo "1) 🖥️  Local Development (Free, for testing)"
    echo "2) 🐳 Docker VPS Deployment ($20-30/month)"
    echo "3) ☁️  AWS Production Deployment ($100-150/month)"
    echo ""
    read -p "Enter choice [1-3]: " choice
    
    case $choice in
        1) DEPLOYMENT_TYPE="local" ;;
        2) DEPLOYMENT_TYPE="docker" ;;
        3) DEPLOYMENT_TYPE="aws" ;;
        *) print_error "Invalid choice"; exit 1 ;;
    esac
}

# Function for local deployment
deploy_local() {
    print_status "Setting up local development environment..."
    
    # Check if Python is installed
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if pip is installed
    if ! command_exists pip3; then
        print_status "Installing pip..."
        sudo apt update && sudo apt install -y python3-pip
    fi
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip3 install -r requirements.txt
    
    # Start Redis container
    print_status "Starting Redis container..."
    docker run -d --name factr-ai-redis -p 6379:6379 redis:7-alpine
    
    # Create local environment file
    print_status "Creating local