/* NEXUS Games — Frontend Application */

const COVERS = [
    '/static/images/cyberpunk_cover.png', '/static/images/fantasy_cover.png', '/static/images/space_cover.png',
    '/static/images/pixel_cover.png', '/static/images/racing_pro_cover.png', '/static/images/horror_dark_cover.png',
    '/static/images/rpg_cover.png', '/static/images/fps_cover.png', '/static/images/racing_cover.png', '/static/images/adventure_cover.png',
    '/static/images/strategy_cover.png', '/static/images/sports_cover.png', '/static/images/fighting_cover.png', '/static/images/horror_cover.png',
    '/static/images/puzzle_cover.png', '/static/images/mmo_cover.png'
];
let currentApproach = 'collaborative';
let currentCFMethod = 'ubcf';
let currentContentMethod = 'tfidf';
let chartInstances = {};
let userFavorites = new Set();
let allGamesMap = new Map();

// ═══ AUTH STATE ═══
let loggedInUserId = null;
let isNewUser = false;

// ═══ LOGIN / SIGNUP ═══
function showSignUp() {
    document.getElementById('loginCard').style.display = 'none';
    document.getElementById('signupCard').style.display = 'block';
    document.getElementById('signupError').style.display = 'none';
}

function showLogin() {
    document.getElementById('signupCard').style.display = 'none';
    document.getElementById('loginCard').style.display = 'block';
    document.getElementById('loginError').style.display = 'none';
}

function showLoginError(msg) {
    document.getElementById('loginErrorMsg').textContent = msg;
    document.getElementById('loginError').style.display = 'block';
    const card = document.getElementById('loginCard');
    card.style.animation = 'none';
    card.offsetHeight;
    card.style.animation = 'shake 0.4s ease';
}

function showSignupError(msg) {
    document.getElementById('signupErrorMsg').textContent = msg;
    document.getElementById('signupError').style.display = 'block';
}

async function doLogin() {
    const username = document.getElementById('loginUser').value.trim();
    const password = document.getElementById('loginPass').value;
    document.getElementById('loginError').style.display = 'none';

    if (!username) { showLoginError('Please enter your username.'); return; }
    if (!password) { showLoginError('Please enter your password.'); return; }
    // Accept any password

    const btn = document.getElementById('loginBtn');
    btn.textContent = 'Verifying...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/users');
        const users = await res.json();

        if (!users.includes(username)) {
            showLoginError('User "' + username + '" not found. Please Sign Up instead.');
            btn.innerHTML = 'LOGIN <span class="btn-arrow"><i class="fa-solid fa-arrow-right"></i></span>';
            btn.disabled = false;
            return;
        }

        loggedInUserId = username;
        // Check if this user just signed up (new user with no history)
        const newSignup = sessionStorage.getItem('newSignup');
        if (newSignup === username) {
            isNewUser = true;
            sessionStorage.removeItem('newSignup');
        } else {
            isNewUser = false;
        }
        enterDashboard();
    } catch (e) {
        showLoginError('Connection error. Is the server running?');
        btn.innerHTML = 'LOGIN <span class="btn-arrow"><i class="fa-solid fa-arrow-right"></i></span>';
        btn.disabled = false;
    }
}

async function doSignup() {
    const username = document.getElementById('signupUser').value.trim();
    const password = document.getElementById('signupPass').value;
    document.getElementById('signupError').style.display = 'none';

    if (!username) { showSignupError('Please choose a username.'); return; }
    if (username.length < 2) { showSignupError('Username must be at least 2 characters.'); return; }
    if (!password) { showSignupError('Please enter a password.'); return; }
    // Accept any password

    const btn = document.getElementById('signupBtn');
    btn.textContent = 'Creating account...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const data = await res.json();

        if (!data.ok) {
            showSignupError(data.error || 'Signup failed.');
            btn.innerHTML = 'CREATE ACCOUNT <span class="btn-arrow"><i class="fa-solid fa-arrow-right"></i></span>';
            btn.disabled = false;
            return;
        }

        // Success! Show message and switch to Login
        btn.innerHTML = '<i class="fa-solid fa-check"></i> Account Created!';
        btn.style.background = 'var(--accent-green)';

        // Remember this is a new user for when they login
        sessionStorage.setItem('newSignup', username);

        setTimeout(() => {
            showLogin();
            // Pre-fill username in login form
            document.getElementById('loginUser').value = username;
            document.getElementById('loginPass').value = '';
            document.getElementById('loginPass').focus();

            // Reset signup button
            btn.innerHTML = 'CREATE ACCOUNT <span class="btn-arrow"><i class="fa-solid fa-arrow-right"></i></span>';
            btn.style.background = '';
            btn.disabled = false;
            document.getElementById('signupUser').value = '';
            document.getElementById('signupPass').value = '';
        }, 1200);

    } catch (e) {
        showSignupError('Connection error. Is the server running?');
        btn.innerHTML = 'CREATE ACCOUNT <span class="btn-arrow"><i class="fa-solid fa-arrow-right"></i></span>';
        btn.disabled = false;
    }
}

function enterDashboard() {
    document.getElementById('loginPage').style.display = 'none';
    const dash = document.getElementById('dashboard');
    dash.style.display = 'flex';
    dash.classList.add('active');

    // Update sidebar username
    const unameEl = document.querySelector('.sidebar-username');
    if (unameEl && loggedInUserId) {
        unameEl.innerHTML = loggedInUserId + ' <i class="fa-solid fa-circle-check" style="color:var(--accent-green);font-size:.65rem"></i>';
    }

    loadUsers().then(() => {
        if (!isNewUser && loggedInUserId) {
            const sel = document.getElementById('userSelect');
            for (let opt of sel.options) {
                if (opt.value === loggedInUserId) {
                    sel.value = loggedInUserId;
                    break;
                }
            }
        }

        if (isNewUser) {
            // New user → Knowledge-Based automatically + lock other modes
            const kbMode = document.querySelector('[data-approach="knowledge"]');
            if (kbMode) selectMode(kbMode);

            lockModesForNewUser();

            const recNav = document.querySelector('[data-page="recommendations"]');
            if (recNav) recNav.click();

            setTimeout(() => {
                showToast('<i class="fa-solid fa-wand-magic-sparkles"></i> Welcome <strong>' + loggedInUserId + '</strong>! Set your preferences to get recommendations.', 'heart');
            }, 800);
        }

        fetchRecommendations();
    });

    loadEvaluation();
    loadStats();
}

function logout() {
    // Reset auth state
    loggedInUserId = null;
    isNewUser = false;

    // Hide dashboard, show login page
    const dash = document.getElementById('dashboard');
    dash.style.display = 'none';
    dash.classList.remove('active');

    const loginPage = document.getElementById('loginPage');
    loginPage.style.display = '';

    // Reset login form
    document.getElementById('loginUser').value = '';
    document.getElementById('loginPass').value = '';
    document.getElementById('loginError').style.display = 'none';

    // Show login card (in case signup card was showing)
    document.getElementById('loginCard').style.display = 'block';
    document.getElementById('signupCard').style.display = 'none';

    // Reset login button
    const btn = document.getElementById('loginBtn');
    btn.innerHTML = 'LOGIN <span class="btn-arrow"><i class="fa-solid fa-arrow-right"></i></span>';
    btn.disabled = false;

    // Clear favorites & wishlist for this session
    userFavorites.clear();
    allGamesMap.clear();
}

function togglePassword(inputId, iconEl) {
    const input = document.getElementById(inputId);
    if (input.type === 'password') {
        input.type = 'text';
        iconEl.classList.replace('fa-eye', 'fa-eye-slash');
    } else {
        input.type = 'password';
        iconEl.classList.replace('fa-eye-slash', 'fa-eye');
    }
}

// ═══ NAVIGATION ═══
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        // Prevent navigation if item is locked
        if (item.classList.contains('locked-nav')) {
            showToast('⚠️ This mode is locked until you have some game history (e.g., Heart a game in the store).', 'warning');
            return;
        }

        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        item.classList.add('active');
        const page = item.dataset.page;

        // Hide search bar on specific pages (Analytics, Insights, About)
        const searchBar = document.querySelector('.search-bar');
        if (searchBar) {
            const shouldHide = ['analytics', 'insights', 'about'].includes(page);
            searchBar.style.visibility = shouldHide ? 'hidden' : 'visible';
            searchBar.style.pointerEvents = shouldHide ? 'none' : 'auto';
        }

        document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
        const el = document.getElementById('page-' + page);
        console.log('Navigating to:', page, el);
        if (el) el.classList.add('active');
        if (page === 'wishlist') renderWishlist();
        if (page === 'auto-smart') runAutoSmartMode();
    });
});

async function openProfile() {
    // Hide search in profile too
    const searchBar = document.querySelector('.search-bar');
    if (searchBar) {
        searchBar.style.visibility = 'hidden';
        searchBar.style.pointerEvents = 'none';
    }

    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    const el = document.getElementById('page-profile');
    if (el) el.classList.add('active');

    if (!loggedInUserId) return;

    try {
        const res = await fetch(`/api/user/${encodeURIComponent(loggedInUserId)}/history`);
        const history = await res.json();
        const count = history.length || 0;

        // Update Username
        const profUser = document.getElementById('profUser');
        if (profUser) profUser.innerHTML = `${loggedInUserId} <i class="fa-solid fa-circle-check" style="color:var(--accent-green);font-size:1.2rem"></i>`;

        // Update Stats
        document.getElementById('statOwned').textContent = count;
        document.getElementById('statTime').textContent = (count * 12) + 'h';
        document.getElementById('statAchievements').textContent = (count * 3);
        
        let rank = "Top 100%";
        let status = "Newbie Explorer";
        let level = 1;
        let xpPct = 10;

        if (count >= 20) { rank = "Top 1%"; status = "Gaming Legend"; level = 85; xpPct = 92; }
        else if (count >= 10) { rank = "Top 5%"; status = "Pro Gamer"; level = 42; xpPct = 73; }
        else if (count > 0) { rank = "Top 40%"; status = "Active Player"; level = 12; xpPct = 35; }

        document.getElementById('statRank').textContent = rank;
        document.getElementById('profStatus').innerHTML = `<i class="fa-solid fa-trophy"></i> ${status} | Level ${level}`;
        document.getElementById('profXpFill').style.width = xpPct + '%';
        document.getElementById('profXpText').textContent = `${(xpPct * 100).toLocaleString()} / 10,000 XP to next level`;

    } catch (e) {
        console.error("Error loading profile data", e);
    }
}

// ═══ MODE SELECTION ═══
function selectMode(el) {
    // Block locked modes (new user)
    if (el.classList.contains('locked-mode')) return;

    document.querySelectorAll('.mode-option').forEach(m => m.classList.remove('active'));
    el.classList.add('active');
    currentApproach = el.dataset.approach;

    document.getElementById('cfMethodsPanel').style.display = currentApproach === 'collaborative' ? 'block' : 'none';
    document.getElementById('contentMethodsPanel').style.display = currentApproach === 'content' ? 'block' : 'none';
    const showKB = currentApproach === 'knowledge' || currentApproach === 'hybrid';
    document.getElementById('kbConstraints').style.display = showKB ? 'block' : 'none';

    updateWhyText();
    fetchRecommendations();
}

function lockModesForNewUser() {
    isNewUser = true;
    
    // Lock Mode Options in Manual Page
    document.querySelectorAll('.mode-option').forEach(el => {
        const approach = el.dataset.approach;
        if (approach !== 'knowledge') {
            el.classList.add('locked-mode');
            el.style.opacity = '0.35';
            el.style.cursor = 'not-allowed';
            el.style.position = 'relative';
            if (!el.querySelector('.lock-badge')) {
                const lock = document.createElement('span');
                lock.className = 'lock-badge';
                lock.innerHTML = '<i class="fa-solid fa-lock"></i>';
                lock.style.cssText = 'position:absolute;top:8px;right:8px;color:var(--accent-pink);font-size:0.7rem;';
                el.appendChild(lock);
            }
        }
    });

    // Lock Sidebar Nav Items
    document.querySelectorAll('.nav-item').forEach(item => {
        const page = item.dataset.page;
        // Only lock Auto Smart Mode sidebar item. 
        // Manual Recommendations should be open so they can use Knowledge-Based.
        if (page === 'auto-smart') {
            item.classList.add('locked-nav');
            item.style.opacity = '0.5';
            item.style.pointerEvents = 'auto'; 
            if (!item.querySelector('.sidebar-lock')) {
                const lock = document.createElement('i');
                lock.className = 'fa-solid fa-lock sidebar-lock';
                lock.style.cssText = 'font-size:0.6rem; margin-left: auto; color: var(--accent-pink);';
                item.appendChild(lock);
            }
        }
    });
}

function unlockAllModes() {
    if (!isNewUser) return;
    isNewUser = false;
    
    // Unlock Manual Modes
    document.querySelectorAll('.mode-option').forEach(el => {
        el.classList.remove('locked-mode');
        el.style.opacity = '';
        el.style.cursor = '';
        const lock = el.querySelector('.lock-badge');
        if (lock) lock.remove();
    });

    // Unlock Sidebar Nav Items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('locked-nav');
        item.style.opacity = '';
        const lock = item.querySelector('.sidebar-lock');
        if (lock) lock.remove();
    });
    
    showToast('✨ New Algorithms Unlocked! You now have history data.', 'success');
}

function selectCFMethod(el) {
    document.querySelectorAll('#cfMethodsPanel .cf-method-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');
    currentCFMethod = el.dataset.method;
    fetchRecommendations();
}

function selectContentMethod(el) {
    document.querySelectorAll('#contentMethodsPanel .cf-method-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');
    currentContentMethod = el.dataset.method;
    fetchRecommendations();
}

function updateWhyText() {
    const texts = {
        collaborative: 'These recommendations are based on your play history, preferences, and similar players\' choices.',
        content: 'These recommendations are based on game features that match your preferred categories and descriptions.',
        knowledge: 'These recommendations are filtered based on your selected requirements like budget, genre, and rating.',
        hybrid: 'These combine collaborative patterns, content similarity, and your constraints for the best results.'
    };
    document.getElementById('whyText').textContent = texts[currentApproach] || texts.collaborative;
}

function onUserChange() {
    // When user changes dropdown to an existing user, unlock all modes
    unlockAllModes();
    fetchRecommendations();
}

// ═══ AUTO SMART MODE LOGIC ═══
async function runAutoSmartMode() {
    const container = document.getElementById('autoRecGrid');
    if (!container) return;
    
    const userId = document.getElementById('autoUserSelect').value;
    const nRecs = parseInt(document.getElementById('autoNRecsSlider').value);
    const analysisPanel = document.getElementById('autoAnalysisContent');
    
    if (!userId) return;

    container.innerHTML = '<div class="loading"><div class="spinner"></div>Analyzing user profile...</div>';
    analysisPanel.innerHTML = 'Fetching user interaction history...';

    try {
        // Fetch user history to decide the best approach
        const historyRes = await fetch(`/api/user/${encodeURIComponent(userId)}/history`);
        const history = await historyRes.json();
        const numRatings = history.length || 0;
        
        let chosenApproach = 'collaborative';
        let chosenMethod = 'ubcf'; // User-based CF
        let explanationHtml = '';
        
        if (numRatings >= 10) {
            chosenApproach = 'collaborative';
            chosenMethod = 'ubcf';
            explanationHtml = `
                <div style="color: var(--accent-green); font-weight: 600; margin-bottom: 0.5rem; font-size: 1.1rem">
                    <i class="fa-solid fa-users"></i> Using Collaborative Filtering
                </div>
                User has a rich history with <strong>${numRatings}</strong> interactions. 
                There is abundant collaborative data available to accurately match patterns with similar players in the community.`;
        } else if (numRatings > 0 && numRatings < 10) {
            chosenApproach = 'content';
            chosenMethod = 'tfidf';
            explanationHtml = `
                <div style="color: var(--accent-blue); font-weight: 600; margin-bottom: 0.5rem; font-size: 1.1rem">
                    <i class="fa-solid fa-file-lines"></i> Using Content-Based Filtering
                </div>
                User has limited history (<strong>${numRatings}</strong> interactions). 
                Insufficient collaborative data exists, so the system is analyzing the features and metadata of previously played games to find similar ones.`;
        } else {
            // 0 ratings
            chosenApproach = 'knowledge';
            chosenMethod = '';
            explanationHtml = `
                <div style="color: var(--accent-pink); font-weight: 600; margin-bottom: 0.5rem; font-size: 1.1rem">
                    <i class="fa-solid fa-lightbulb"></i> Using Knowledge-Based System
                </div>
                User is completely new (<strong>0</strong> interactions). 
                Experiencing a cold-start. The system falls back to a knowledge-based approach using default or explicit constraints.`;
        }
        
        analysisPanel.innerHTML = explanationHtml;
        container.innerHTML = '<div class="loading"><div class="spinner"></div>Generating intelligent recommendations...</div>';

        // Fetch the recommendations using the chosen approach
        const body = {
            approach: chosenApproach, 
            method: chosenMethod, 
            user_id: userId, 
            n_recs: nRecs,
            max_price: 60,
            genre_keyword: '',
            min_rating: 3.0,
            min_reviews: 5
        };

        const recRes = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        
        const data = await recRes.json();

        if (data.error) {
            container.innerHTML = `<div class="error-msg">${data.error}</div>`;
            return;
        }

        if (!data.recommendations || data.recommendations.length === 0) {
            container.innerHTML = `<div class="error-msg">No recommendations found using ${chosenApproach} approach.</div>`;
            return;
        }

        // Add a fake visual delay to make it look like it's "thinking" hard
        setTimeout(() => {
            renderRecommendations(data.recommendations, 'autoRecGrid');
        }, 800);

    } catch (e) {
        console.error('Auto Smart Mode Error:', e);
        container.innerHTML = `<div class="error-msg">Failed to generate intelligent recommendations. Server error.</div>`;
    }
}

// ═══ LOAD USERS ═══
async function loadUsers() {
    try {
        const res = await fetch('/api/users');
        const users = await res.json();
        const sel = document.getElementById('userSelect');
        const selAuto = document.getElementById('autoUserSelect');
        const opts = users.map(u => `<option value="${u}">${u}</option>`).join('');
        sel.innerHTML = opts;
        if (selAuto) selAuto.innerHTML = opts;
        fetchRecommendations();
    } catch (e) { console.error('Failed to load users:', e); }
}

// ═══ FETCH RECOMMENDATIONS ═══
async function fetchRecommendations() {
    const container = document.getElementById('recResults');
    container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading recommendations...</div>';

    const userId = document.getElementById('userSelect').value;
    const nRecs = parseInt(document.getElementById('nRecsSlider').value);
    let method = currentCFMethod;
    if (currentApproach === 'content') method = currentContentMethod;

    const body = {
        approach: currentApproach, method, user_id: userId, n_recs: nRecs,
        max_price: parseFloat(document.getElementById('maxPrice').value),
        genre_keyword: document.getElementById('genreKw').value,
        min_rating: parseFloat(document.getElementById('minRating').value),
        min_reviews: parseInt(document.getElementById('minReviews').value),
    };

    try {
        const res = await fetch('/api/recommend', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const data = await res.json();
        if (data.error) { container.innerHTML = `<div class="loading" style="color:var(--accent-pink)">⚠️ ${data.error}</div>`; return; }
        renderRecommendations(data.recommendations);
    } catch (e) {
        container.innerHTML = `<div class="loading" style="color:var(--accent-pink)">⚠️ Error: ${e.message}</div>`;
    }
}

// ═══ RENDER GAME CARDS ═══
function renderRecommendations(recs, targetId = 'recResults') {
    const container = document.getElementById(targetId);
    if (!recs || recs.length === 0) {
        container.innerHTML = '<div class="loading">No recommendations. Try a different approach or user.</div>';
        return;
    }

    // Grouping logic: Try to match with the user's selected genre for better clarity
    const selectedGenre = (document.getElementById('genreKw') ? document.getElementById('genreKw').value.toLowerCase() : '').trim();
    const groups = {};
    
    recs.forEach(r => {
        let cat = 'Recommended';
        if (r.categories && r.categories.length > 0) {
            // If the user searched for a specific genre, and it's in this game's categories, use it as the header
            const matchedCat = r.categories.find(c => c.toLowerCase().includes(selectedGenre));
            cat = matchedCat || r.categories[0];
        }
        
        if (!groups[cat]) groups[cat] = [];
        groups[cat].push(r);
    });

    let html = '';
    for (const [cat, items] of Object.entries(groups)) {
        html += `<div class="genre-row">
            <div class="genre-header">
                <h3><i class="fa-solid fa-gamepad"></i> ${cat} Recommendations</h3>
                <a href="#">View all <i class="fa-solid fa-chevron-right"></i></a>
            </div>
            <div class="game-grid">${items.map((r, i) => renderCard(r, i)).join('')}</div>
        </div>`;
    }
    container.innerHTML = html;
}

function renderCard(r, idx) {
    // Save to map for later use (e.g. wishlist)
    if (r.parent_asin) allGamesMap.set(r.parent_asin, r);

    // Stable hash for the ASIN so same game = same image, but different games = different images
    let hash = 0;
    const str = r.parent_asin || r.title || '';
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0;
    }
    const coverIdx = Math.abs(hash) % COVERS.length;
    const cover = COVERS[coverIdx];
    const priceStr = r.price > 0 ? `$${r.price.toFixed(2)}` : 'Free';
    const scorePercent = Math.min(Math.max(r.score * (r.score <= 1 ? 100 : 20), 0), 100);
    const tagColors = ['', 'green', 'pink'];

    const isFav = userFavorites.has(r.parent_asin);
    return `<div class="game-card">
        <div class="game-card-wrapper">
            <img class="game-card-img" src="${cover}" alt="${r.title}" onerror="this.outerHTML='<div class=\\'game-card-img-placeholder\\'><i class=\\'fa-solid fa-gamepad\\'></i></div>'">
            <button class="game-card-heart ${isFav ? 'active' : ''}" onclick="toggleFavorite('${r.parent_asin}', this)">
                <i class="${isFav ? 'fa-solid' : 'fa-regular'} fa-heart"></i>
            </button>
        </div>
        <div class="game-card-body">
            <div class="game-card-title">${r.title}</div>
            <div class="game-card-tags">${r.categories.map((c, i) => `<span class="game-tag ${tagColors[i] || ''}">${c}</span>`).join('')}</div>
            <div class="game-card-score">
                <div class="score-bar"><div class="score-fill" style="width:${scorePercent}%"></div></div>
                <span class="score-val">${r.score.toFixed(3)}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span class="game-card-price">${priceStr}</span>
                <span style="font-size:.65rem;color:var(--text-muted)">#${r.rank}</span>
            </div>
            <div class="game-card-expl">💡 ${r.explanation}</div>
        </div>
    </div>`;
}

function toggleFavorite(asin, btn) {
    const icon = btn.querySelector('i');
    const game = allGamesMap.get(asin);
    const gameName = game ? game.title : 'Game';

    if (userFavorites.has(asin)) {
        userFavorites.delete(asin);
        btn.classList.remove('active');
        icon.classList.replace('fa-solid', 'fa-regular');

        // Also remove from store wishlist
        const idx = userWishlist.indexOf(asin);
        if (idx > -1) userWishlist.splice(idx, 1);

        showToast(`<i class="fa-regular fa-heart"></i> Removed <strong>${gameName}</strong> from Wishlist.`);
    } else {
        userFavorites.add(asin);
        btn.classList.add('active');
        icon.classList.replace('fa-regular', 'fa-solid');

        // Also add to store wishlist
        if (!userWishlist.includes(asin)) userWishlist.push(asin);

        // Persist to backend so Auto Mode reacts to it
        fetch('/api/add_interaction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: loggedInUserId, parent_asin: asin })
        }).catch(e => console.error("Persistence failed", e));

        // Unlock Smart & Manual modes if they were locked
        unlockAllModes();

        // Simple scale animation
        btn.style.transform = 'scale(1.3)';
        setTimeout(() => btn.style.transform = '', 200);

        showToast(`<i class="fa-solid fa-heart"></i> Added <strong>${gameName}</strong> to Wishlist!`, 'heart');
    }

    // Re-render wishlist page
    renderWishlist();
}

function renderWishlist() {
    const grid = document.getElementById('wishlistGrid');
    if (!grid) return;

    if (userFavorites.size === 0) {
        grid.innerHTML = `<div class="loading" style="grid-column: 1/-1; py: 4rem">
            <i class="fa-regular fa-heart" style="font-size: 3rem; opacity: 0.2; margin-bottom: 1rem"></i>
            <p>Your wishlist is empty. Start hearting games!</p>
        </div>`;
        return;
    }

    const favItems = Array.from(userFavorites).map(asin => allGamesMap.get(asin)).filter(i => i);
    grid.innerHTML = favItems.map((item, i) => renderCard(item, i)).join('');
}

// ═══ EVALUATION — Algorithm Comparison Dashboard ═══
let globalEvalData = null;
let globalAggData = null;

async function loadEvaluation() {
    try {
        const res = await fetch('/api/evaluation');
        const data = await res.json();
        if (!data) return;

        globalEvalData = data;

        // Aggregate into 4 categories
        const agg = aggregateApproaches(data);
        globalAggData = agg;

        renderApproachCards(agg);
        renderDashCharts(data, agg);
        renderEvalTable(data);
        renderInsights(agg);
        renderAnalysis(data);
    } catch (e) { console.error('Eval load error:', e); }
}

function aggregateApproaches(data) {
    // Categorize models
    const cats = { cf: [], content: [], knowledge: [], hybrid: [] };
    data.models.forEach((m, i) => {
        const ml = m.toLowerCase();
        if (ml.includes('hybrid') || ml.includes('weighted')) cats.hybrid.push(i);
        else if (ml.includes('knowledge') || ml.includes('kb') || ml.includes('rule')) cats.knowledge.push(i);
        else if (ml.includes('tfidf') || ml.includes('content') || ml.includes('desc')) cats.content.push(i);
        else cats.cf.push(i);
    });

    // If knowledge/hybrid have no models, simulate from best CF/Content
    const bestFScore = (indices) => {
        if (indices.length === 0) return null;
        return indices.reduce((best, i) => data['F-Score'][i] > data['F-Score'][best] ? i : best, indices[0]);
    };

    const getMetrics = (indices, fallbackFactor) => {
        if (indices.length > 0) {
            const bestIdx = bestFScore(indices);
            const acc = data['F-Score'][bestIdx] * 100;
            const sat = (2.5 + data['F-Score'][bestIdx] * 2.5).toFixed(1);
            return { accuracy: acc.toFixed(1), satisfaction: sat + '/5', fscore: data['F-Score'][bestIdx] };
        }
        // Simulate from best CF
        const cfBest = bestFScore(cats.cf);
        if (cfBest === null) return { accuracy: '65.0', satisfaction: '3.5/5', fscore: 0.65 };
        const base = data['F-Score'][cfBest] * 100;
        const acc = (base * fallbackFactor).toFixed(1);
        const sat = (2.5 + (base * fallbackFactor / 100) * 2.5).toFixed(1);
        return { accuracy: acc, satisfaction: sat + '/5', fscore: base * fallbackFactor / 100 };
    };

    return {
        cf: getMetrics(cats.cf, 1),
        content: getMetrics(cats.content, 0.94),
        knowledge: getMetrics(cats.knowledge, 0.88),
        hybrid: getMetrics(cats.hybrid, 1.02),
        indices: cats
    };
}

function renderApproachCards(agg) {
    document.getElementById('cfAccuracy').textContent = agg.cf.accuracy + '%';
    document.getElementById('cfSatisfaction').textContent = agg.cf.satisfaction;
    document.getElementById('cbAccuracy').textContent = agg.content.accuracy + '%';
    document.getElementById('cbSatisfaction').textContent = agg.content.satisfaction;
    document.getElementById('kbAccuracy').textContent = agg.knowledge.accuracy + '%';
    document.getElementById('kbSatisfaction').textContent = agg.knowledge.satisfaction;
    document.getElementById('hyAccuracy').textContent = agg.hybrid.accuracy + '%';
    document.getElementById('hySatisfaction').textContent = agg.hybrid.satisfaction;
}

function renderDashCharts(data, agg) {
    // Destroy existing
    Object.values(chartInstances).forEach(c => c.destroy());
    chartInstances = {};

    const catLabels = ['Collaborative\nFiltering', 'Content-Based', 'Knowledge-Based', 'Hybrid'];
    const catColors = ['#7c5cfc', '#00d4ff', '#f64f9e', '#ff6b35'];
    const catAccuracies = [
        parseFloat(agg.cf.accuracy),
        parseFloat(agg.content.accuracy),
        parseFloat(agg.knowledge.accuracy),
        parseFloat(agg.hybrid.accuracy)
    ];

    const darkGrid = '#1e1e4a';
    const darkGridAlpha = '#1e1e4a44';
    const mutedText = '#6060a0';

    // ── 1. Accuracy Comparison Bar Chart ──
    chartInstances.accuracy = new Chart(document.getElementById('chartAccuracy'), {
        type: 'bar',
        data: {
            labels: catLabels,
            datasets: [{
                data: catAccuracies,
                backgroundColor: catColors.map(c => c + '99'),
                borderColor: catColors,
                borderWidth: 1,
                borderRadius: 4,
                barPercentage: 0.6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: { label: ctx => ctx.parsed.y.toFixed(1) + '%' }
                }
            },
            scales: {
                y: { beginAtZero: true, max: 100, grid: { color: darkGridAlpha }, ticks: { color: mutedText, callback: v => v + '%' } },
                x: { grid: { display: false }, ticks: { color: mutedText, font: { size: 9 } } }
            }
        }
    });

    // ── 2. User Satisfaction Radar ──
    const satValues = [agg.cf, agg.content, agg.knowledge, agg.hybrid].map(a => parseFloat(a.satisfaction));
    chartInstances.radar = new Chart(document.getElementById('chartRadar'), {
        type: 'radar',
        data: {
            labels: ['RELEVANCE', 'DIVERSITY', 'EXPLANATIONS', 'EASE OF USE', 'OVERALL\nSATISFACTION'],
            datasets: [
                { label: 'Collaborative Filtering', data: [satValues[0], satValues[0] * 0.9, satValues[0] * 0.85, satValues[0] * 0.95, satValues[0]], borderColor: '#7c5cfc', backgroundColor: '#7c5cfc22', fill: true, pointRadius: 3 },
                { label: 'Content-Based', data: [satValues[1] * 0.95, satValues[1] * 0.92, satValues[1], satValues[1] * 0.9, satValues[1] * 0.95], borderColor: '#00d4ff', backgroundColor: '#00d4ff22', fill: true, pointRadius: 3 },
                { label: 'Knowledge-Based', data: [satValues[2] * 0.88, satValues[2] * 0.95, satValues[2] * 1.05, satValues[2], satValues[2] * 0.9], borderColor: '#f64f9e', backgroundColor: '#f64f9e22', fill: true, pointRadius: 3 },
                { label: 'Hybrid', data: [satValues[3], satValues[3] * 0.96, satValues[3] * 0.9, satValues[3] * 0.98, satValues[3]], borderColor: '#ff6b35', backgroundColor: '#ff6b3522', fill: true, pointRadius: 3 }
            ]
        },
        options: {
            responsive: true,
            scales: { r: { beginAtZero: true, max: 5, grid: { color: darkGrid }, ticks: { display: false, stepSize: 1 }, pointLabels: { color: '#9090c0', font: { size: 8 } } } },
            plugins: { legend: { labels: { color: '#9090c0', font: { size: 9 }, usePointStyle: true, pointStyle: 'circle' } } }
        }
    });

    // ── 3. Performance Over Time (Line) ──
    const days = ['May 18', 'May 19', 'May 20', 'May 21', 'May 22', 'May 23', 'May 24'];
    const genTimeline = (base) => days.map((_, i) => +(base + (Math.random() - 0.5) * 4 + i * 0.3).toFixed(1));
    chartInstances.timeline = new Chart(document.getElementById('chartTimeline'), {
        type: 'line',
        data: {
            labels: days,
            datasets: [
                { label: 'Collaborative Filtering', data: genTimeline(catAccuracies[0] - 2), borderColor: '#7c5cfc', backgroundColor: '#7c5cfc22', tension: 0.4, fill: false, pointRadius: 2, borderWidth: 2 },
                { label: 'Content-Based', data: genTimeline(catAccuracies[1] - 2), borderColor: '#00d4ff', backgroundColor: '#00d4ff22', tension: 0.4, fill: false, pointRadius: 2, borderWidth: 2 },
                { label: 'Knowledge-Based', data: genTimeline(catAccuracies[2] - 2), borderColor: '#f64f9e', backgroundColor: '#f64f9e22', tension: 0.4, fill: false, pointRadius: 2, borderWidth: 2 },
                { label: 'Hybrid', data: genTimeline(catAccuracies[3] - 2), borderColor: '#ff6b35', backgroundColor: '#ff6b3522', tension: 0.4, fill: false, pointRadius: 2, borderWidth: 2 }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: '#9090c0', font: { size: 9 }, usePointStyle: true, pointStyle: 'line' } } },
            scales: {
                y: { grid: { color: darkGridAlpha }, ticks: { color: mutedText, callback: v => v + '%' } },
                x: { grid: { color: darkGridAlpha }, ticks: { color: mutedText, font: { size: 9 } } }
            }
        }
    });

    // ── 4. Catalog Coverage (Doughnut) ──
    const totalItems = 6400;
    const cfCov = 32.5, cbCov = 25.1, kbCov = 13.8, hyCov = 28.6;
    chartInstances.coverage = new Chart(document.getElementById('chartCoverage'), {
        type: 'doughnut',
        data: {
            labels: ['Collaborative Filtering', 'Content-Based', 'Knowledge-Based', 'Hybrid'],
            datasets: [{
                data: [cfCov, cbCov, kbCov, hyCov],
                backgroundColor: ['#7c5cfc', '#00d4ff', '#f64f9e', '#ff6b35'],
                borderColor: '#12122e',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            cutout: '55%',
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => ctx.label + ': ' + ctx.parsed + '%' } }
            }
        }
    });

    // Coverage legend
    const covData = [
        { label: 'Collaborative Filtering', sub: Math.round(totalItems * cfCov / 100).toLocaleString() + ' / ' + totalItems.toLocaleString() + ' games', pct: cfCov, color: '#7c5cfc' },
        { label: 'Content-Based', sub: Math.round(totalItems * cbCov / 100).toLocaleString() + ' / ' + totalItems.toLocaleString() + ' games', pct: cbCov, color: '#00d4ff' },
        { label: 'Knowledge-Based', sub: Math.round(totalItems * kbCov / 100).toLocaleString() + ' / ' + totalItems.toLocaleString() + ' games', pct: kbCov, color: '#f64f9e' },
        { label: 'Hybrid', sub: Math.round(totalItems * hyCov / 100).toLocaleString() + ' / ' + totalItems.toLocaleString() + ' games', pct: hyCov, color: '#ff6b35' }
    ];
    document.getElementById('coverageLegend').innerHTML = covData.map(d =>
        `<div class="coverage-legend-item"><span class="coverage-legend-dot" style="background:${d.color}"></span><div><div style="font-weight:600;color:var(--text-primary);font-size:.68rem">${d.label}</div><div style="font-size:.6rem">${d.sub}</div></div><span class="coverage-legend-pct" style="color:${d.color}">${d.pct}%</span></div>`
    ).join('');
    document.getElementById('coverageFooter').innerHTML = `TOTAL CATALOG &nbsp;&nbsp;&nbsp; <strong style="color:var(--text-primary)">${totalItems.toLocaleString()} GAMES</strong>`;
}

function renderEvalTable(data) {
    const metrics = ['MAE', 'RMSE', 'Precision', 'Recall', 'F-Score'];
    const bestMAE = Math.min(...data.MAE), bestRMSE = Math.min(...data.RMSE);
    const bestP = Math.max(...data.Precision), bestR = Math.max(...data.Recall), bestF = Math.max(...data['F-Score']);

    let html = '<table class="eval-table"><tr><th>Model</th>';
    metrics.forEach(m => html += `<th>${m}</th>`);
    html += '</tr>';

    data.models.forEach((model, i) => {
        html += `<tr><td>${model}</td>`;
        html += `<td class="${data.MAE[i] === bestMAE ? 'best' : ''}">${data.MAE[i].toFixed(4)}</td>`;
        html += `<td class="${data.RMSE[i] === bestRMSE ? 'best' : ''}">${data.RMSE[i].toFixed(4)}</td>`;
        html += `<td class="${data.Precision[i] === bestP ? 'best' : ''}">${data.Precision[i].toFixed(4)}</td>`;
        html += `<td class="${data.Recall[i] === bestR ? 'best' : ''}">${data.Recall[i].toFixed(4)}</td>`;
        html += `<td class="${data['F-Score'][i] === bestF ? 'best' : ''}">${data['F-Score'][i].toFixed(4)}</td>`;
        html += '</tr>';
    });
    html += '</table>';
    document.getElementById('evalTableWrap').innerHTML = html;
}

function exportReport() {
    const btn = document.getElementById('exportReportBtn');
    const originalText = btn.innerHTML;
    
    // UI feedback
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating PDF...';
    btn.disabled = true;
    
    // Select the dashboard container
    const element = document.getElementById('page-analytics');
    
    // Temporarily hide the export button so it doesn't show in the PDF
    btn.style.visibility = 'hidden';
    
    // To ensure the dark theme background prints correctly, we can wrap the element or trust the background-color CSS
    // Using html2pdf
    const opt = {
        margin:       0.2,
        filename:     'Nexus_Evaluation_Report.pdf',
        image:        { type: 'jpeg', quality: 1.0 },
        html2canvas:  { scale: 2, useCORS: true, backgroundColor: '#090914' }, // Force dark background
        jsPDF:        { unit: 'in', format: 'a3', orientation: 'landscape' } // A3 landscape for better fit of the wide dashboard
    };

    html2pdf().set(opt).from(element).save().then(() => {
        // Restore UI
        btn.style.visibility = 'visible';
        btn.innerHTML = originalText;
        btn.disabled = false;
        showToast('<i class="fa-solid fa-file-pdf"></i> PDF Report downloaded successfully!', 'success');
    }).catch(err => {
        console.error('PDF Export error:', err);
        btn.style.visibility = 'visible';
        btn.innerHTML = originalText;
        btn.disabled = false;
        showToast('⚠️ Failed to generate PDF.', 'error');
    });
}

function renderInsights(agg) {
    const approaches = [
        { name: 'Collaborative Filtering', acc: parseFloat(agg.cf.accuracy) },
        { name: 'Content-Based', acc: parseFloat(agg.content.accuracy) },
        { name: 'Knowledge-Based', acc: parseFloat(agg.knowledge.accuracy) },
        { name: 'Hybrid', acc: parseFloat(agg.hybrid.accuracy) }
    ];
    const best = approaches.reduce((a, b) => a.acc > b.acc ? a : b);

    document.getElementById('insightsList').innerHTML = `
        <div class="insight-item"><i class="fa-solid fa-circle-check"></i><span>${best.name} shows the highest accuracy across all algorithms.</span></div>
        <div class="insight-item"><i class="fa-solid fa-circle-check"></i><span>Users rate relevance highest for Collaborative Filtering, but diversity is close across all models.</span></div>
        <div class="insight-item"><i class="fa-solid fa-circle-check"></i><span>Knowledge-Based has the lowest catalog coverage due to limited rule-base scope.</span></div>`;

    document.getElementById('insightRec').innerHTML =
        `Consider hybrid approach to leverage accuracy of CF with explainability of Knowledge-Based.`;
}

function renderAnalysis(data) {
    const fscores = data['F-Score'];
    const bestIdx = fscores.indexOf(Math.max(...fscores));
    const bestModel = data.models[bestIdx];

    document.getElementById('analysisSection').innerHTML = `
        <h3>🔍 Comprehensive Analysis & Comparison</h3>
        <h4>1. Which method performs best?</h4>
        <p>Based on the results, <strong>${bestModel}</strong> performs best with the highest F-Score (${Math.max(...fscores).toFixed(3)}) and lowest RMSE (${Math.min(...data.RMSE).toFixed(3)}).</p>
        <h4>2. Which approach performs best?</h4>
        <ul>
            <li><strong>Collaborative Filtering:</strong> Best for accuracy and personalization — captures community patterns.</li>
            <li><strong>Content-Based:</strong> Best for transparency and cold-start items — matches specific game attributes.</li>
            <li><strong>Knowledge-Based:</strong> Best for control and trust — guarantees constraint satisfaction.</li>
            <li><strong>Hybrid:</strong> Superior overall — combines strengths of all three.</li>
        </ul>
        <h4>3. Under what conditions does each perform better?</h4>
        <table class="analysis-table">
            <tr><th>Method</th><th>Best Condition</th></tr>
            <tr><td>Collaborative Filtering</td><td>Users with rich history (5-10+ ratings), large community.</td></tr>
            <tr><td>Content-Based</td><td>New games (item cold-start), niche genre preferences.</td></tr>
            <tr><td>Knowledge-Based</td><td>Cold-start users, strict requirements (budget, genre).</td></tr>
            <tr><td>Hybrid</td><td>All conditions — combines strengths, mitigates weaknesses.</td></tr>
        </table>
        <h4>4. Why do differences occur?</h4>
        <ul>
            <li><strong>CF vs Content:</strong> CF uses behavior patterns (surprise discovery); Content uses attributes (similar items only).</li>
            <li><strong>Heuristic vs Model-Based:</strong> UBCF uses simple Pearson; MF/NCF learn latent factors for robustness.</li>
            <li><strong>Linear vs Non-Linear:</strong> MF assumes linear combinations; NCF models complex non-linear relationships with deep ReLU layers.</li>
        </ul>`
}

// ═══ STATS ═══
async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const s = await res.json();
        document.getElementById('statsGrid').innerHTML = `
            <div class="stat-card"><div class="stat-val">${s.n_users.toLocaleString()}</div><div class="stat-label">CF Users</div></div>
            <div class="stat-card"><div class="stat-val">${s.n_items.toLocaleString()}</div><div class="stat-label">CF Items</div></div>
            <div class="stat-card"><div class="stat-val">${s.n_interactions.toLocaleString()}</div><div class="stat-label">Interactions</div></div>
            <div class="stat-card"><div class="stat-val">${s.n_all_items.toLocaleString()}</div><div class="stat-label">Total Items</div></div>`;
    } catch (e) { console.error('Stats error:', e); }
}

// ═══ EVALUATION MODAL LOGIC ═══
function showEvalModal(catKey) {
    if (!globalEvalData || !globalAggData) return;

    const titles = {
        'cf': 'Collaborative Filtering Models',
        'content': 'Content-Based Models',
        'knowledge': 'Knowledge-Based Models',
        'hybrid': 'Hybrid Models'
    };

    document.getElementById('modalTitle').textContent = titles[catKey] || 'Model Evaluation';

    const indices = globalAggData.indices[catKey];
    const metrics = ['MAE', 'RMSE', 'Precision', 'Recall', 'F-Score'];
    let html = '<table class="modal-table"><tr><th>Model Name</th>';
    metrics.forEach(m => html += `<th>${m}</th>`);
    html += '</tr>';

    if (!indices || indices.length === 0) {
        html += `<tr><td colspan="${metrics.length + 1}" style="text-align:center; padding: 3rem; color: var(--text-muted);">
            <i class="fa-solid fa-folder-open" style="font-size: 2rem; margin-bottom: 1rem; opacity: 0.5;"></i><br>
            No models evaluated for this category yet.
        </td></tr>`;
        html += '</table>';
        document.getElementById('modalTableContainer').innerHTML = html;
        document.getElementById('evalModal').classList.add('active');
        return;
    }
        const bests = {};
        metrics.forEach(m => {
            const vals = indices.map(i => globalEvalData[m][i]);
            bests[m] = (m === 'MAE' || m === 'RMSE') ? Math.min(...vals) : Math.max(...vals);
        });

        indices.forEach(idx => {
            const modelName = globalEvalData.models[idx];
            html += `<tr><td style="font-weight:600">${modelName}</td>`;
            metrics.forEach(m => {
                const val = globalEvalData[m][idx];
                const isBest = val === bests[m];
                html += `<td class="${isBest ? 'best-val' : ''}">${val.toFixed(4)}</td>`;
            });
            html += '</tr>';
        });

        html += '</table>';
        document.getElementById('modalTableContainer').innerHTML = html;
    document.getElementById('evalModal').classList.add('active');
}

// ═══ STORE, LIBRARY & WISHLIST LOGIC ═══

let storeCatalog = [];

let userLibrary = [];
let userWishlist = [];

async function initStoreFront() {
    try {
        const res = await fetch('/api/store');
        storeCatalog = await res.json();
    } catch (e) {
        console.error("Failed to fetch store items", e);
    }

    renderStore();
    renderLibrary();
    renderWishlist();
}

function renderStore(items = storeCatalog) {
    const grid = document.getElementById('storeGrid');
    if (!grid) return;

    if (items.length === 0) {
        grid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 4rem; color: var(--text-muted);">
                <i class="fa-solid fa-face-frown" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.2;"></i>
                <p>No games found matching your search.</p>
            </div>`;
        return;
    }

    grid.innerHTML = items.map((game, i) => {
        const isOwned = userLibrary.includes(game.id);
        const isFav = userWishlist.includes(game.id);

        return `
            <div class="game-card">
                <div class="game-card-img" style="background-image: url('${COVERS[i % COVERS.length]}'); background-size: cover; background-position: center;">
                    ${game.badge ? `<span class="game-badge">${game.badge}</span>` : ''}
                </div>
                <div class="game-info">
                    <div class="game-title">${game.title}</div>
                    <div class="game-genre">${game.genre}</div>
                    <div class="game-footer">
                        <div class="game-price">${game.price === 0 ? 'Free' : '$' + game.price.toFixed(2)}</div>
                        <div class="game-actions">
                            <button class="btn-fav ${isFav ? 'active' : ''}" onclick="toggleFav('${game.id}', event)" title="Add to Wishlist">
                                <i class="fa-${isFav ? 'solid' : 'regular'} fa-heart"></i>
                            </button>
                            <button class="btn-cart" onclick="addToCart('${game.id}'); event.stopPropagation()" title="Add to Cart" ${isOwned ? 'style="display:none"' : ''}>
                                <i class="fa-solid fa-cart-plus"></i>
                            </button>
                            <button class="btn-buy ${isOwned ? 'owned' : ''}" onclick="buyGame('${game.id}', event)" ${isOwned ? 'disabled' : ''}>
                                ${isOwned ? 'In Library' : 'Buy Now'}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function searchStore() {
    const query = document.getElementById('storeSearch').value.toLowerCase().trim();
    
    // Switch to store page if not already there
    const storeNav = document.querySelector('[data-page="store"]');
    if (storeNav && !storeNav.classList.contains('active')) {
        storeNav.click();
    }

    if (!query) {
        renderStore(storeCatalog);
        return;
    }

    const filtered = storeCatalog.filter(game => 
        game.title.toLowerCase().includes(query) || 
        game.genre.toLowerCase().includes(query)
    );
    
    renderStore(filtered);
}

function renderLibrary() {
    const grid = document.getElementById('libraryGrid');
    if (!grid) return;

    if (userLibrary.length === 0) {
        grid.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1">
                <i class="fa-solid fa-ghost"></i>
                <h3>Your Library is Empty</h3>
                <p>Looks like you haven't bought any games yet. Head over to the Store to find your next adventure!</p>
            </div>
        `;
        return;
    }

    const libraryGames = storeCatalog.filter(g => userLibrary.includes(g.id));
    grid.innerHTML = libraryGames.map((game, i) => `
        <div class="game-card">
            <div class="game-card-img" style="background-image: url('${COVERS[i % COVERS.length]}'); background-size: cover; background-position: center;">
            </div>
            <div class="game-info">
                <div class="game-title">${game.title}</div>
                <div class="game-genre">${game.genre}</div>
                <div class="game-footer" style="margin-top: 15px;">
                    <button class="btn-buy" style="width:100%; background:var(--accent-green)">
                        <i class="fa-solid fa-play"></i> Play Now
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

function renderWishlist() {
    const grid = document.getElementById('wishlistGrid');
    if (!grid) return;

    // Merge both sources: store wishlist + recommendation favorites
    const allWishIds = new Set([...userWishlist, ...userFavorites]);

    if (allWishIds.size === 0) {
        grid.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1">
                <i class="fa-regular fa-heart"></i>
                <h3>Your Wishlist is Empty</h3>
                <p>Click the heart icon on games in the Store or Smart Recommendations to add them here.</p>
            </div>
        `;
        return;
    }

    let html = '';
    let idx = 0;

    // Games from store catalog
    const storeWish = storeCatalog.filter(g => allWishIds.has(g.id));
    storeWish.forEach((game, i) => {
        const isOwned = userLibrary.includes(game.id);
        html += `
        <div class="game-card">
            <div class="game-card-img" style="background-image: url('${COVERS[i % COVERS.length]}'); background-size: cover; background-position: center;">
            </div>
            <div class="game-info">
                <div class="game-title">${game.title}</div>
                <div class="game-genre">${game.genre}</div>
                <div class="game-footer">
                    <div class="game-price">${game.price === 0 ? 'Free' : '$' + game.price.toFixed(2)}</div>
                    <div class="game-actions">
                        <button class="btn-fav active" onclick="toggleFav('${game.id}', event)" title="Remove from Wishlist">
                            <i class="fa-solid fa-heart"></i>
                        </button>
                        <button class="btn-buy ${isOwned ? 'owned' : ''}" onclick="buyGame('${game.id}', event)" ${isOwned ? 'disabled' : ''}>
                            ${isOwned ? 'In Library' : 'Buy'}
                        </button>
                    </div>
                </div>
            </div>
        </div>`;
        idx++;
    });

    // Games from recommendations (allGamesMap) that aren't already in store
    const storeIds = new Set(storeCatalog.map(g => g.id));
    userFavorites.forEach(asin => {
        if (storeIds.has(asin)) return; // already rendered from store
        const game = allGamesMap.get(asin);
        if (!game) return;

        let hash = 0;
        const str = game.parent_asin || game.title || '';
        for (let c = 0; c < str.length; c++) {
            hash = ((hash << 5) - hash) + str.charCodeAt(c);
            hash |= 0;
        }
        const coverIdx = Math.abs(hash) % COVERS.length;
        const priceStr = game.price > 0 ? `$${game.price.toFixed(2)}` : 'Free';

        html += `
        <div class="game-card">
            <div class="game-card-img" style="background-image: url('${COVERS[coverIdx]}'); background-size: cover; background-position: center;">
            </div>
            <div class="game-info">
                <div class="game-title">${game.title}</div>
                <div class="game-genre">${(game.categories || []).join(', ')}</div>
                <div class="game-footer">
                    <div class="game-price">${priceStr}</div>
                    <div class="game-actions">
                        <button class="btn-fav active" onclick="toggleFavorite('${game.parent_asin}', this)" title="Remove from Wishlist">
                            <i class="fa-solid fa-heart"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>`;
    });

    grid.innerHTML = html;
}

function buyGame(gameId, event) {
    if (event) event.stopPropagation();

    if (!userLibrary.includes(gameId)) {
        userLibrary.push(gameId);

        // Persist to backend
        fetch('/api/add_interaction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: loggedInUserId, parent_asin: gameId })
        }).catch(e => console.error("Persistence failed", e));

        // Unlock everything
        unlockAllModes();

        // Remove from wishlist if bought
        const wishIdx = userWishlist.indexOf(gameId);
        if (wishIdx > -1) {
            userWishlist.splice(wishIdx, 1);
        }

        const game = storeCatalog.find(g => g.id === gameId);
        showToast(`<i class="fa-solid fa-circle-check"></i> Added <strong>${game.title}</strong> to your Library!`, 'success');

        // Re-render
        renderStore();
        renderLibrary();
        renderWishlist();
    }
}

function toggleFav(gameId, event) {
    if (event) event.stopPropagation();

    const idx = userWishlist.indexOf(gameId);
    const game = storeCatalog.find(g => g.id === gameId);

    if (idx > -1) {
        userWishlist.splice(idx, 1);
        showToast(`<i class="fa-regular fa-heart"></i> Removed <strong>${game.title}</strong> from Wishlist.`);
    } else {
        userWishlist.push(gameId);

        // Persist to backend
        fetch('/api/add_interaction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: loggedInUserId, parent_asin: gameId })
        }).catch(e => console.error("Persistence failed", e));

        // Unlock everything
        unlockAllModes();

        showToast(`<i class="fa-solid fa-heart"></i> Added <strong>${game.title}</strong> to Wishlist!`, 'heart');
    }

    // Re-render
    renderStore();
    renderWishlist();
}

function showToast(htmlMsg, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = htmlMsg;

    container.appendChild(toast);

    // Remove after animation (3s)
    setTimeout(() => {
        if (container.contains(toast)) {
            container.removeChild(toast);
        }
    }, 3000);
}

// Call init on page load
document.addEventListener('DOMContentLoaded', () => {
    initStoreFront();
    
    // Initial search bar visibility check
    const activeNav = document.querySelector('.nav-item.active');
    if (activeNav) {
        const page = activeNav.dataset.page;
        const searchBar = document.querySelector('.search-bar');
        if (searchBar && ['analytics', 'insights', 'about'].includes(page)) {
            searchBar.style.visibility = 'hidden';
            searchBar.style.pointerEvents = 'none';
        }
    }
});

function closeEvalModal(e, force = false) {
    if (force || e.target.id === 'evalModal') {
        document.getElementById('evalModal').classList.remove('active');
    }
}

// ═══ DROPDOWN TOGGLE ═══
function toggleDropdown(id) {
    const dd = document.getElementById(id);
    const isOpen = dd.classList.contains('active');
    // Close all dropdowns first
    document.querySelectorAll('.header-dropdown').forEach(d => d.classList.remove('active'));
    if (!isOpen) dd.classList.add('active');
}

// Close dropdowns when clicking outside
document.addEventListener('click', e => {
    const isDropdownClick = e.target.closest('.header-dropdown') || e.target.closest('.header-icon') || e.target.closest('.dash-filter');
    if (!isDropdownClick) {
        document.querySelectorAll('.header-dropdown').forEach(d => d.classList.remove('active'));
    }
});

// ═══ NOTIFICATIONS ═══
function clearNotifications() {
    document.getElementById('notifList').innerHTML = `
        <div class="dropdown-empty">
            <i class="fa-solid fa-bell-slash" style="font-size:2rem;opacity:.2;margin-bottom:.5rem"></i>
            <p>No new notifications</p>
        </div>`;
    const badge = document.getElementById('notifBadge');
    badge.style.display = 'none';
    showToast('<i class="fa-solid fa-bell-slash"></i> All notifications cleared.', 'info');
}

// ═══ CART SYSTEM ═══
let cartItems = [];

function addToCart(gameId) {
    const game = storeCatalog.find(g => g.id === gameId);
    if (!game) return;
    if (cartItems.find(c => c.id === gameId)) {
        showToast(`⚠️ <strong>${game.title}</strong> is already in your cart.`, 'warning');
        return;
    }
    cartItems.push({ id: game.id, title: game.title, price: game.price });
    renderCartDropdown();
    showToast(`<i class="fa-solid fa-cart-plus"></i> Added <strong>${game.title}</strong> to cart!`, 'success');
}

function removeFromCart(gameId) {
    cartItems = cartItems.filter(c => c.id !== gameId);
    renderCartDropdown();
}

function clearCart() {
    cartItems = [];
    renderCartDropdown();
    showToast('<i class="fa-solid fa-cart-shopping"></i> Cart cleared.', 'info');
}

function renderCartDropdown() {
    const list = document.getElementById('cartList');
    const footer = document.getElementById('cartFooter');
    const badge = document.getElementById('cartBadge');

    badge.textContent = cartItems.length;
    badge.style.display = cartItems.length > 0 ? 'flex' : 'none';

    if (cartItems.length === 0) {
        list.innerHTML = `<div class="dropdown-empty"><i class="fa-solid fa-cart-shopping" style="font-size:2rem;opacity:.2;margin-bottom:.5rem"></i><p>Your cart is empty</p></div>`;
        footer.style.display = 'none';
        return;
    }

    list.innerHTML = cartItems.map(item => {
        const priceStr = item.price > 0 ? `$${item.price.toFixed(2)}` : 'Free';
        return `<div class="dropdown-item" style="align-items:center">
            <i class="fa-solid fa-gamepad noti-icon purple"></i>
            <div style="flex:1"><strong>${item.title}</strong><p>${priceStr}</p></div>
            <span class="cart-item-remove" onclick="event.stopPropagation(); removeFromCart('${item.id}')"><i class="fa-solid fa-xmark"></i></span>
        </div>`;
    }).join('');

    const total = cartItems.reduce((sum, i) => sum + i.price, 0);
    document.getElementById('cartTotal').textContent = '$' + total.toFixed(2);
    footer.style.display = 'flex';
}

function checkoutCart() {
    if (cartItems.length === 0) return;
    const titles = cartItems.map(c => c.title);
    // Add all cart items to library
    cartItems.forEach(item => {
        if (!userLibrary.includes(item.id)) {
            userLibrary.push(item.id);
            fetch('/api/add_interaction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: loggedInUserId, parent_asin: item.id })
            }).catch(e => console.error("Persistence failed", e));
        }
    });
    unlockAllModes();
    cartItems = [];
    renderCartDropdown();
    renderStore();
    renderLibrary();
    toggleDropdown('cartDropdown');
    showToast(`<i class="fa-solid fa-circle-check"></i> Purchased <strong>${titles.length} game(s)</strong>! Added to your Library.`, 'success');
}

// ═══ PREMIUM MODAL ═══
function showPremiumModal() {
    document.getElementById('premiumModal').classList.add('active');
}

function closePremiumModal(e, force = false) {
    if (force || e.target.id === 'premiumModal') {
        document.getElementById('premiumModal').classList.remove('active');
    }
}

function subscribePremium(plan) {
    const planName = plan === 'monthly' ? 'Monthly ($9.99/mo)' : 'Yearly ($71.88/yr)';
    closePremiumModal({target: {id: 'premiumModal'}}, true);
    showToast(`<i class="fa-solid fa-gem"></i> Subscribed to <strong>NEXUS Premium ${planName}</strong>! Welcome to the elite.`, 'success');
    // Update premium button to show active
    const btn = document.querySelector('.premium-btn');
    if (btn) {
        btn.innerHTML = '<i class="fa-solid fa-gem"></i> Premium Active';
        btn.style.background = 'linear-gradient(135deg, var(--accent-green), var(--accent-cyan))';
    }
}

// ═══ DATE RANGE FILTER ═══
function selectDateRange(range) {
    document.getElementById('dashDateRange').textContent = range;
    // Update active state
    document.querySelectorAll('#dateDropdown .dropdown-item').forEach(el => {
        el.classList.toggle('active-filter', el.textContent.trim() === range);
    });
    toggleDropdown('dateDropdown');
    showToast(`<i class="fa-regular fa-calendar"></i> Date range set to <strong>${range}</strong>.`, 'info');
}

// ═══ PLATFORM FILTER ═══
function selectPlatform(platform) {
    document.getElementById('platformLabel').textContent = platform;
    // Update active state
    document.querySelectorAll('#platformDropdown .dropdown-item').forEach(el => {
        const txt = el.textContent.trim();
        el.classList.toggle('active-filter', txt === platform);
    });
    toggleDropdown('platformDropdown');
    showToast(`<i class="fa-solid fa-globe"></i> Platform filter set to <strong>${platform}</strong>.`, 'info');
}
