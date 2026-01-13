#!/bin/bash

# Deployment script for Horror Games Review Intelligence Blog
# Usage: ./deploy.sh [method]
# Methods: docker, apache, nginx, static

set -e

DEPLOY_METHOD=${1:-docker}
PROJECT_DIR=$(pwd)

echo "🎮 Horror Games Review Intelligence - Deployment Script"
echo "======================================================"
echo ""

case $DEPLOY_METHOD in
  docker)
    echo "🐳 Deploying with Docker..."
    echo ""

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Build and run
    echo "Building Docker image..."
    docker build -t horror-games-blog .

    echo "Starting container..."
    docker-compose up -d

    echo ""
    echo "✅ Deployment complete!"
    echo "🌐 Site available at: http://localhost:8080"
    echo ""
    echo "Commands:"
    echo "  - Stop:    docker-compose down"
    echo "  - Logs:    docker-compose logs -f"
    echo "  - Restart: docker-compose restart"
    ;;

  apache)
    echo "🔧 Deploying to Apache..."
    echo ""

    # Default Apache web root (adjust as needed)
    APACHE_ROOT=${2:-/var/www/html/horror-games-blog}

    echo "Creating directory: $APACHE_ROOT"
    sudo mkdir -p "$APACHE_ROOT"

    echo "Copying files..."
    sudo cp -r "$PROJECT_DIR"/* "$APACHE_ROOT/"

    echo "Setting permissions..."
    sudo chown -R www-data:www-data "$APACHE_ROOT"
    sudo chmod -R 755 "$APACHE_ROOT"

    echo "Enabling mod_rewrite..."
    sudo a2enmod rewrite
    sudo a2enmod headers
    sudo a2enmod expires

    echo "Restarting Apache..."
    sudo systemctl restart apache2

    echo ""
    echo "✅ Deployment complete!"
    echo "🌐 Site should be available at your Apache server"
    ;;

  nginx)
    echo "🔧 Deploying to Nginx..."
    echo ""

    # Default Nginx web root (adjust as needed)
    NGINX_ROOT=${2:-/var/www/horror-games-blog}

    echo "Creating directory: $NGINX_ROOT"
    sudo mkdir -p "$NGINX_ROOT"

    echo "Copying files..."
    sudo cp -r "$PROJECT_DIR"/* "$NGINX_ROOT/"

    echo "Setting permissions..."
    sudo chown -R www-data:www-data "$NGINX_ROOT"
    sudo chmod -R 755 "$NGINX_ROOT"

    echo ""
    echo "Sample Nginx configuration:"
    echo "======================================"
    cat << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    root /var/www/horror-games-blog;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    gzip on;
    gzip_types text/css text/html application/javascript;
}
EOF
    echo "======================================"
    echo ""
    echo "Save this configuration to: /etc/nginx/sites-available/horror-games-blog"
    echo "Then run:"
    echo "  sudo ln -s /etc/nginx/sites-available/horror-games-blog /etc/nginx/sites-enabled/"
    echo "  sudo nginx -t"
    echo "  sudo systemctl restart nginx"
    ;;

  static)
    echo "📦 Creating static deployment package..."
    echo ""

    PACKAGE_NAME="horror-games-blog-$(date +%Y%m%d-%H%M%S).tar.gz"

    echo "Creating archive: $PACKAGE_NAME"
    tar -czf "$PACKAGE_NAME" \
        --exclude="*.sh" \
        --exclude="Dockerfile" \
        --exclude="docker-compose.yml" \
        --exclude=".git" \
        .

    echo ""
    echo "✅ Package created: $PACKAGE_NAME"
    echo ""
    echo "Upload this file to your hosting provider:"
    echo "  - GitHub Pages: Extract and push to gh-pages branch"
    echo "  - Netlify: Drag and drop the extracted folder"
    echo "  - Vercel: Run 'vercel --prod' in the extracted folder"
    ;;

  *)
    echo "❌ Unknown deployment method: $DEPLOY_METHOD"
    echo ""
    echo "Available methods:"
    echo "  - docker:  Deploy using Docker (default)"
    echo "  - apache:  Deploy to Apache server"
    echo "  - nginx:   Deploy to Nginx server"
    echo "  - static:  Create deployment package"
    echo ""
    echo "Usage: ./deploy.sh [method]"
    exit 1
    ;;
esac
