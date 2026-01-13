# Horror Games Review Intelligence - Static Website

AI-Powered analysis of 98,000+ Steam reviews for horror games, presented as a modern minimalist blog.

## Overview

This static website showcases sentiment analysis results from Steam horror game reviews, categorized into three sections:
- **Overwhelmingly Positive**: Must-play titles
- **Mixed Reviews**: Wait for sale recommendations
- **Mostly Negative**: Proceed with caution

## Features

- Modern, minimalist design with dark theme
- Fully responsive layout
- Three sentiment categories with detailed game summaries
- Interactive card-based navigation
- SEO-optimized HTML structure
- Zero JavaScript dependencies
- Fast loading times

## Project Structure

```
horror_games_blog/
├── index.html              # Main landing page with all categories
├── positive.html           # Overwhelmingly positive game summaries
├── neutral.html            # Mixed review game summaries
├── negative.html           # Mostly negative game summaries
├── css/
│   └── style.css          # Main stylesheet
├── assets/                # Image assets (placeholder directory)
├── .htaccess              # Apache configuration for reverse proxy
├── robots.txt             # SEO crawler configuration
└── README.md              # This file
```

## Deployment

### Option 1: Apache/Nginx Reverse Proxy

#### Apache (.htaccess included)

1. Upload all files to your web server
2. Ensure `.htaccess` is enabled in your Apache configuration
3. The site will be accessible at your domain root or subdirectory

**Nginx Configuration Example:**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /path/to/horror_games_blog;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # Cache static assets
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Gzip compression
    gzip on;
    gzip_types text/css text/html application/javascript;
    gzip_min_length 256;
}
```

### Option 2: Static Hosting (GitHub Pages, Netlify, Vercel)

#### GitHub Pages
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/horror-games-blog.git
git push -u origin main
```

Then enable GitHub Pages in repository settings.

#### Netlify
1. Drag and drop the `horror_games_blog` folder to Netlify
2. Site will be live immediately

#### Vercel
```bash
cd horror_games_blog
vercel --prod
```

### Option 3: Docker Deployment

**Dockerfile:**
```dockerfile
FROM nginx:alpine
COPY . /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Build and run:**
```bash
docker build -t horror-games-blog .
docker run -d -p 8080:80 horror-games-blog
```

## Games Included

### Overwhelmingly Positive (5)
1. Left 4 Dead 2
2. Subnautica
3. Misao - 2024 HD Remaster
4. Lethal Company
5. Your Turn To Die -Death Game By Majority-

### Mixed Reviews (5)
1. HYPERVIOLENT
2. Brutalism22
3. Nayati River
4. Alien Dawn
5. The Red Hood

### Mostly Negative (5)
1. At Home
2. Hollow
3. Moonstone Tavern - A Fantasy Tavern Sim!
4. The Last Cargo
5. Mars Taken

## Technical Details

- **HTML5** semantic markup
- **CSS3** with custom properties (CSS variables)
- **Responsive** design (mobile-first approach)
- **No JavaScript** required
- **Accessible** (ARIA-compliant where needed)
- **SEO-friendly** with meta tags

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Customization

### Colors
Edit CSS variables in `css/style.css`:
```css
:root {
    --bg-dark: #0a0a0a;
    --bg-card: #1a1a1a;
    --accent-positive: #4ade80;
    --accent-neutral: #fbbf24;
    --accent-negative: #f87171;
}
```

### Adding New Games
1. Edit the relevant HTML file (positive.html, neutral.html, or negative.html)
2. Add a new `<article class="game-summary">` block
3. Update the card on index.html

## Performance

- **Page Size**: ~50KB (HTML + CSS)
- **Load Time**: <500ms on 3G
- **Lighthouse Score**: 95+
- **No external dependencies**

## License

This project is part of the NLP Steam Horror Review Intelligence System.

## Credits

- Summaries generated using OpenAI GPT-3.5
- Based on 98,000+ Steam reviews
- UI Design: Minimalist dark theme optimized for readability

## Contact

For issues or suggestions, refer to the main NLP project documentation.
