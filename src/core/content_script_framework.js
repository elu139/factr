    extractText(postElement) {
        // Enhanced text extraction for posts and reels
        const textSelectors = [
            // Instagram feed posts
            'div[data-testid="post-text"]',
            'span[dir="auto"]',  // Common for Instagram text
            'div > span',        // Generic span within divs
            
            // Instagram reels
            'h1',  // Reel titles sometimes use h1
            'div[role="button"] span',  // Interactive reel text
            
            // Fallback selectors
            'p', 'div', 'span'
        ];
        
        // Try each selector in order of specificity
        for (const selector of textSelectors) {
            const elements = postElement.querySelectorAll(selector);
            
            for (const element of elements) {
                const text = element.textContent.trim();
                
                // Filter out likely non-content text
                if (this.isValidPostText(text)) {
                    return text;
                }
            }
        }
        
        // Final fallback: get all text but clean it
        const allText = postElement.textContent.trim();
        return this.cleanExtractedText(allText);
    }
    
    extractText(postElement) {
        // Enhanced text extraction for posts and reels with validation
        const textSelectors = [
            // Instagram feed posts (priority order)
            'span[dir="auto"]',          // Most common for Instagram text
            'div[data-testid*="post"] span',  // Post-specific spans
            'h1',                        // Reel titles
            'div[role="button"] span',   // Interactive elements
            
            // Fallback selectors  
            'p', 'div > span', 'span'
        ];
        
        let extractedTexts = [];
        
        // Try each selector and collect valid text
        for (const selector of textSelectors) {
            const elements = postElement.querySelectorAll(selector);
            
            for (const element of elements) {
                const text = element.textContent.trim();
                
                if (this.isValidPostText(text)) {
                    extractedTexts.push(text);
                }
            }
        }
        
        // Find the longest valid text (likely the main content)
        if (extractedTexts.length > 0) {
            const mainText = extractedTexts.reduce((longest, current) => 
                current.length > longest.length ? current : longest
            );
            return this.cleanExtractedText(mainText);
        }
        
        // Final fallback: clean all text
        const allText = postElement.textContent.trim();
        return this.cleanExtractedText(allText);
    }
    
    isValidPostText(text) {
        // Enhanced validation for post content
        if (!text || text.length < 10) return false;
        
        // Skip common UI elements and metadata
        const uiPatterns = [
            /^(?:like|comment|share|save)$/i,
            /^(?:follow|following|followers?)$/i,
            /^(?:\d+[kmb]?\s*(?:likes?|views?|comments?))$/i,
            /^(?:ago|hours?|days?|weeks?)\s*$/i,
            /^(?:verified|official)$/i,
            /^@\w+$/,  // Just a mention
            /^#\w+$/,  // Just a hashtag
            /^\d+$/,   // Just numbers
            /^[•·\-\s]+$/,  // Just punctuation/separators
        ];
        
        // Check if text matches UI patterns
        for (const pattern of uiPatterns) {
            if (pattern.test(text.trim())) {
                return false;
            }
        }
        
        // Must have some meaningful content
        const meaningfulWords = text.split(/\s+/).filter(word => 
            word.length > 2 && !/^[@#]\w+$/.test(word)
        );
        
        return meaningfulWords.length >= 3;
    }
    
    cleanExtractedText(text) {
        // Clean and normalize extracted text
        if (!text) return '';
        
        // Remove excessive whitespace
        text = text.replace(/\s+/g, ' ').trim();
        
        // Remove obvious UI artifacts
        text = text.replace(/^(?:Like|Comment|Share|Save)\s*/gi, '');
        text = text.replace(/\s*(?:\d+[kmb]?\s*(?:likes?|views?|comments?))$/gi, '');
        
        // Limit length to prevent processing issues
        if (text.length > 2000) {
            text = text.substring(0, 2000) + '...';
        }
        
        return text.trim();
    }
    
    extractImages(postElement) {
        // Enhanced image extraction with quality filtering
        const images = postElement.querySelectorAll('img');
        const validImages = [];
        
        Array.from(images).forEach(img => {
            const src = img.src;
            
            // Skip if no source
            if (!src) return;
            
            // Skip common UI images
            const skipPatterns = [
                /avatar/i,
                /profile/i,
                /icon/i,
                /sprite/i,
                /emoji/i,
                /stories_tray/i,
                /placeholder/i
            ];
            
            const shouldSkip = skipPatterns.some(pattern => pattern.test(src));
            if (shouldSkip) return;
            
            // Check image dimensions (avoid tiny images)
            const width = img.naturalWidth || img.width || 0;
            const height = img.naturalHeight || img.height || 0;
            
            // Only include reasonably sized images (likely content, not UI)
            if (width >= 150 && height >= 150) {
                validImages.push(src);
            }
        });
        
        // Limit to first 3 images to avoid processing overload
        return validImages.slice(0, 3);
    }
    
    extractAuthor(postElement) {
        // Enhanced author extraction with multiple strategies
        const authorSelectors = [
            // Instagram-specific author patterns
            'header a[href*="/"]',           // Profile links in headers
            'a[role="link"][href*="/"] span', // Link spans with profile refs
            'h2 span',                       // Header spans
            'div[data-testid*="user"] span', // User test IDs
            
            // Generic patterns
            'a[href*="/"] span',
            'strong', 'b',                   // Bold text (often usernames)
            'h1', 'h2', 'h3'                // Headers
        ];
        
        for (const selector of authorSelectors) {
            const elements = postElement.querySelectorAll(selector);
            
            for (const element of elements) {
                const text = element.textContent.trim();
                
                // Validate author text
                if (this.isValidAuthorText(text)) {
                    return this.cleanAuthorText(text);
                }
            }
        }
        
        return 'Unknown';
    }
    
    isValidAuthorText(text) {
        // Validate potential author text
        if (!text || text.length < 2 || text.length > 50) return false;
        
        // Skip obvious non-author text
        const skipPatterns = [
            /^(?:like|comment|share|save|follow|following)$/i,
            /^(?:\d+\s*(?:likes?|views?|comments?|followers?))$/i,
            /^(?:verified|official|ago|hours?|days?)$/i,
            /^[@#]/,  // Mentions/hashtags aren't authors
            /[{}()[\]]/,  // Code-like text
            /^\d+$/   // Just numbers
        ];
        
        return !skipPatterns.some(pattern => pattern.test(text));
    }
    
    cleanAuthorText(text) {
        // Clean author text
        return text.replace(/^@/, '').trim();  // Remove @ prefix if present
    }
    
    extractTimestamp(postElement) {
        // Enhanced timestamp extraction
        const timeSelectors = [
            'time',
            '[datetime]',
            'a[href*="/p/"] time',  // Post-specific time links
            '[title*="at"]',
            'span[title]'           // Spans with title attributes
        ];
        
        for (const selector of timeSelectors) {
            const elements = postElement.querySelectorAll(selector);
            
            for (const element of elements) {
                // Try multiple timestamp sources
                const sources = [
                    element.getAttribute('datetime'),
                    element.getAttribute('title'),
                    element.textContent
                ];
                
                for (const source of sources) {
                    if (source) {
                        const timestamp = this.parseTimestamp(source);
                        if (timestamp) {
                            return timestamp;
                        }
                    }
                }
            }
        }
        
        // Fallback to current time
        return new Date().toISOString();
    }
    
    parseTimestamp(timestampStr) {
        // Parse various timestamp formats
        if (!timestampStr) return null;
        
        try {
            // Try ISO format first
            const isoDate = new Date(timestampStr);
            if (!isNaN(isoDate.getTime())) {
                return isoDate.toISOString();
            }
            
            // Try relative time parsing
            const relativePatterns = [
                { pattern: /(\d+)\s*h(?:our)?s?\s*ago/i, multiplier: 3600000 },
                { pattern: /(\d+)\s*m(?:in(?:ute)?)?s?\s*ago/i, multiplier: 60000 },
                { pattern: /(\d+)\s*d(?:ay)?s?\s*ago/i, multiplier: 86400000 },
                { pattern: /(\d+)\s*w(?:eek)?s?\s*ago/i, multiplier: 604800000 }
            ];
            
            for (const { pattern, multiplier } of relativePatterns) {
                const match = timestampStr.match(pattern);
                if (match) {
                    const amount = parseInt(match[1]);
                    const timestamp = new Date(Date.now() - (amount * multiplier));
                    return timestamp.toISOString();
                }
            }
            
        } catch (error) {
            // Ignore parsing errors
        }
        
        return null;
    }// content.js - Framework for social media post detection
class FactrContentDetector {
    constructor() {
        this.platform = this.detectPlatform();
        this.observer = null;
        this.analyzedPosts = new Set();
        this.init();
    }
    
    detectPlatform() {
        const hostname = window.location.hostname;
        if (hostname.includes('instagram.com')) return 'instagram';
        if (hostname.includes('twitter.com') || hostname.includes('x.com')) return 'twitter';
        if (hostname.includes('facebook.com')) return 'facebook';
        if (hostname.includes('tiktok.com')) return 'tiktok';
        return 'unknown';
    }
    
    init() {
        console.log(`Factr.ai: Initializing on ${this.platform}`);
        
        // Start observing DOM changes for dynamic content
        this.startObserver();
        
        // Analyze existing posts on page
        this.analyzeExistingPosts();
        
        // Listen for messages from popup/background
        chrome.runtime.onMessage.addListener(this.handleMessage.bind(this));
    }
    
    startObserver() {
        this.observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            this.checkForNewPosts(node);
                        }
                    });
                }
            });
        });
        
        this.observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
    
    checkForNewPosts(element) {
        const posts = this.findPosts(element);
        posts.forEach(post => this.processPost(post));
    }
    
    analyzeExistingPosts() {
        const posts = this.findPosts(document);
        console.log(`Found ${posts.length} existing posts`);
        posts.forEach(post => this.processPost(post));
    }
    
    findPosts(container) {
        // Focus on Instagram posts and reels specifically
        const posts = [];
        
        // Method 1: Standard feed posts (articles) - most reliable
        const feedPosts = container.querySelectorAll ? 
            container.querySelectorAll('article') : [];
        
        // Method 2: Look for elements that might be reels or stories
        const reelElements = container.querySelectorAll ? 
            container.querySelectorAll('[role="presentation"]') : [];
        
        // Combine candidates and validate them
        const candidateElements = [...feedPosts, ...reelElements];
        
        candidateElements.forEach(element => {
            // Skip if we've already analyzed this element
            if (element.dataset.factrAnalyzed) {
                return;
            }
            
            // Validate this is actually content worth analyzing
            const validation = this.validatePostElement(element);
            
            if (validation.isValid) {
                // Mark as analyzed to prevent duplicate processing
                element.dataset.factrAnalyzed = 'true';
                posts.push(element);
            }
        });
        
        return posts.filter(post => !this.analyzedPosts.has(this.getPostId(post)));
    }
    
    validatePostElement(element) {
        // Enhanced validation for post elements
        const hasText = this.extractText(element).length > 10;
        const images = this.extractImages(element);
        const hasValidImages = images.length > 0;
        const author = this.extractAuthor(element);
        const hasAuthor = author !== 'Unknown';
        
        // Element must have meaningful content
        const isValid = hasText || (hasValidImages && hasAuthor);
        
        // Additional checks to avoid UI elements
        const elementText = element.textContent.toLowerCase();
        const isLikelyUI = elementText.includes('suggested for you') ||
                          elementText.includes('sponsored') ||
                          elementText.length < 20;
        
        return {
            isValid: isValid && !isLikelyUI,
            hasText,
            hasValidImages,
            hasAuthor,
            imageCount: images.length,
            textLength: hasText ? this.extractText(element).length : 0
        };
    }
    
    getPostId(postElement) {
        // Create unique ID for post to avoid re-analysis
        const text = this.extractText(postElement);
        const images = this.extractImages(postElement);
        return btoa(text.substring(0, 100) + images.length).replace(/[^a-zA-Z0-9]/g, '');
    }
    
    async processPost(postElement) {
        const postId = this.getPostId(postElement);
        
        if (this.analyzedPosts.has(postId)) return;
        this.analyzedPosts.add(postId);
        
        try {
            const postData = this.extractPostData(postElement);
            
            // Validate extracted data
            if (!postData.text && postData.images.length === 0) {
                console.log('Factr.ai: Skipping post with no analyzable content');
                return;
            }
            
            if (postData.text.length < 10 && postData.images.length === 0) {
                console.log('Factr.ai: Skipping post with insufficient content');
                return;
            }
            
            // Add loading indicator
            this.addLoadingIndicator(postElement, postId);
            
            // Send for analysis with timeout handling
            const analysis = await Promise.race([
                this.analyzePost(postData),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Analysis timeout')), 35000)
                )
            ]);
            
            // Display results
            this.displayAnalysis(postElement, analysis, postId);
            
        } catch (error) {
            console.error('Factr.ai: Error processing post:', error);
            this.removeLoadingIndicator(postElement, postId);
            
            // Show error indicator instead of failing silently
            this.displayErrorIndicator(postElement, postId, error.message);
        }
    }
    
    displayErrorIndicator(postElement, postId, errorMessage) {
        // Show a subtle error indicator
        const errorIndicator = document.createElement('div');
        errorIndicator.id = `factr-error-${postId}`;
        errorIndicator.className = 'factr-error-indicator';
        errorIndicator.innerHTML = `
            <div class="factr-error-content">
                <span class="factr-error-icon">⚠️</span>
                <span class="factr-error-text">Analysis unavailable</span>
                <button class="factr-retry-btn" onclick="this.closest('.factr-error-indicator').remove(); window.location.reload();">
                    Retry
                </button>
            </div>
        `;
        
        // Style the error indicator
        errorIndicator.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(239, 68, 68, 0.9);
            color: white;
            padding: 4px 8px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 500;
            z-index: 10000;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        `;
        
        postElement.style.position = 'relative';
        postElement.appendChild(errorIndicator);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            const indicator = document.getElementById(`factr-error-${postId}`);
            if (indicator) {
                indicator.remove();
            }
        }, 10000);
    }
    
    extractPostData(postElement) {
        return {
            text: this.extractText(postElement),
            images: this.extractImages(postElement),
            author: this.extractAuthor(postElement),
            timestamp: this.extractTimestamp(postElement),
            platform: this.platform,
            url: window.location.href
        };
    }
    
    extractText(postElement) {
        // Platform-specific text extraction
        const textSelectors = {
            instagram: ['div[data-testid="post-text"]', 'span', 'div'],
            twitter: ['div[data-testid="tweetText"]', 'span'],
            facebook: ['[data-testid="post_message"]', 'span'],
            tiktok: ['[data-e2e="browse-video-desc"]']
        };
        
        const selectors = textSelectors[this.platform] || ['p', 'span', 'div'];
        
        for (const selector of selectors) {
            const textElement = postElement.querySelector(selector);
            if (textElement && textElement.textContent.trim()) {
                return textElement.textContent.trim();
            }
        }
        
        // Fallback: get all text content
        return postElement.textContent.trim();
    }
    
    extractImages(postElement) {
        const images = postElement.querySelectorAll('img');
        return Array.from(images)
            .map(img => img.src)
            .filter(src => src && !src.includes('avatar') && !src.includes('profile'));
    }
    
    extractAuthor(postElement) {
        // Platform-specific author extraction
        const authorSelectors = {
            instagram: ['h2 span', 'a[role="link"]'],
            twitter: ['div[data-testid="User-Names"] span'],
            facebook: ['h3 a', 'strong a'],
            tiktok: ['[data-e2e="browse-username"]']
        };
        
        const selectors = authorSelectors[this.platform] || ['a', 'span', 'h1', 'h2', 'h3'];
        
        for (const selector of selectors) {
            const authorElement = postElement.querySelector(selector);
            if (authorElement && authorElement.textContent.trim()) {
                return authorElement.textContent.trim();
            }
        }
        
        return 'Unknown';
    }
    
    extractTimestamp(postElement) {
        const timeElements = postElement.querySelectorAll('time, [datetime], [title*="at"]');
        
        for (const timeElement of timeElements) {
            const datetime = timeElement.getAttribute('datetime') || 
                           timeElement.getAttribute('title') || 
                           timeElement.textContent;
            
            if (datetime) {
                const date = new Date(datetime);
                if (!isNaN(date.getTime())) {
                    return date.toISOString();
                }
            }
        }
        
        return new Date().toISOString();
    }
    
    async analyzePost(postData) {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage({
                action: 'analyzePost',
                data: postData
            }, (response) => {
                resolve(response || { error: 'Analysis failed' });
            });
        });
    }
    
    addLoadingIndicator(postElement, postId) {
        const indicator = document.createElement('div');
        indicator.id = `factr-loading-${postId}`;
        indicator.className = 'factr-loading-indicator';
        indicator.innerHTML = `
            <div class="factr-loading-content">
                <div class="factr-spinner"></div>
                <span>Analyzing with Factr.ai...</span>
            </div>
        `;
        
        // Position relative to post
        postElement.style.position = 'relative';
        postElement.appendChild(indicator);
    }
    
    removeLoadingIndicator(postElement, postId) {
        const indicator = document.getElementById(`factr-loading-${postId}`);
        if (indicator) {
            indicator.remove();
        }
    }
    
    displayAnalysis(postElement, analysis, postId) {
        this.removeLoadingIndicator(postElement, postId);
        
        if (analysis.error) {
            console.error('Factr.ai: Analysis error:', analysis.error);
            return;
        }
        
        const overlay = this.createAnalysisOverlay(analysis, postId);
        postElement.appendChild(overlay);
        
        // Store analysis for popup access
        chrome.storage.local.set({
            [`analysis_${postId}`]: analysis
        });
    }
    
    createAnalysisOverlay(analysis, postId) {
        const overlay = document.createElement('div');
        overlay.id = `factr-analysis-${postId}`;
        overlay.className = 'factr-analysis-overlay';
        
        const score = analysis.misinformation_score || 0;
        const riskLevel = this.getRiskLevel(score);
        const color = this.getRiskColor(score);
        
        overlay.innerHTML = `
            <div class="factr-analysis-content" style="border-left: 4px solid ${color}">
                <div class="factr-header">
                    <img src="${chrome.runtime.getURL('icons/icon16.png')}" alt="Factr.ai">
                    <span class="factr-title">Factr.ai Analysis</span>
                    <button class="factr-close" onclick="this.closest('.factr-analysis-overlay').remove()">×</button>
                </div>
                <div class="factr-score">
                    <div class="factr-score-circle" style="border-color: ${color}">
                        <span class="factr-score-value">${score.toFixed(0)}%</span>
                    </div>
                    <div class="factr-score-label">
                        <span class="factr-risk-level" style="color: ${color}">${riskLevel}</span>
                        <span class="factr-confidence">Confidence: ${analysis.confidence_level}</span>
                    </div>
                </div>
                <div class="factr-details">
                    <p class="factr-explanation">${analysis.explanation || 'Analysis completed.'}</p>
                    ${analysis.detected_inconsistencies && analysis.detected_inconsistencies.length > 0 ? `
                        <details class="factr-inconsistencies">
                            <summary>Issues Detected (${analysis.detected_inconsistencies.length})</summary>
                            <ul>
                                ${analysis.detected_inconsistencies.map(issue => `<li>${issue}</li>`).join('')}
                            </ul>
                        </details>
                    ` : ''}
                </div>
            </div>
        `;
        
        return overlay;
    }
    
    getRiskLevel(score) {
        if (score < 20) return 'Low Risk';
        if (score < 40) return 'Moderate Risk';
        if (score < 70) return 'High Risk';
        return 'Very High Risk';
    }
    
    getRiskColor(score) {
        if (score < 20) return '#22c55e'; // Green
        if (score < 40) return '#eab308'; // Yellow
        if (score < 70) return '#f97316'; // Orange
        return '#ef4444'; // Red
    }
    
    handleMessage(message, sender, sendResponse) {
        if (message.action === 'getPageAnalytics') {
            sendResponse({
                platform: this.platform,
                postsAnalyzed: this.analyzedPosts.size,
                url: window.location.href
            });
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new FactrContentDetector();
    });
} else {
    new FactrContentDetector();
}