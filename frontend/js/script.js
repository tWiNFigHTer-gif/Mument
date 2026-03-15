function escapeHTML(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const carData = {
  audi: {
    name: 'Audi R8', brand: 'Audi', price: '$275,000', tag: 'Luxury Cars',
    rating: '4.3', ratingCount: '248', pos: 68, neu: 20, neg: 12,
    img: 'images/audi_r8.png',
    thumbs: [
      'images/audi_r8.png',
      'images/audi_r8_sideview.png',
      'images/audi_r8_frontview.png',
      'images/audi_r8_topview.png'
    ],
    reviews: [
      { name: 'John D', seed: 'John', stars: 5, date: 'Feb 2026', text: 'Amazing performance and design' },
      { name: 'Jaguar G', seed: 'Jaguar', stars: 4, date: 'Jan 2026', text: 'Good but Maintenance is very high' },
      { name: 'Sarah M', seed: 'Sarah', stars: 5, date: 'Jan 2026', text: "Absolutely love the handling and quattro system. Best car I've owned." }
    ]
  },
  punch: {
    name: 'Punch', brand: 'TATA', price: '$115,000', tag: 'SUV',
    rating: '4.3', ratingCount: '248', pos: 75, neu: 8, neg: 17,
    img: 'images/punch.png',
    thumbs: ['images/punch.png'],
    reviews: [
      { name: 'Ravi K', seed: 'Ravi', stars: 5, date: 'Feb 2026', text: 'Excellent value for money. Great build quality!' },
      { name: 'Priya S', seed: 'Priya', stars: 4, date: 'Jan 2026', text: 'Very comfortable ride and excellent mileage. Interior could be better.' }
    ]
  },
  swift: {
    name: 'Swift', brand: 'Suzuki', price: '$90,000', tag: 'Hatchback',
    rating: '4.6', ratingCount: '627', pos: 81, neu: 10, neg: 9,
    img: 'images/swift.png',
    thumbs: ['images/swift.png'],
    reviews: [
      { name: 'Mike T', seed: 'Mike', stars: 5, date: 'Feb 2026', text: 'Best hatchback in its segment! Fun to drive and fuel efficient.' },
      { name: 'Lisa N', seed: 'Lisa', stars: 5, date: 'Jan 2026', text: 'Great car for daily commute. Peppy engine and stylish looks.' }
    ]
  },
  baleno: {
    name: 'Baleno', brand: 'Suzuki', price: '$75,000', tag: 'Hatchback',
    rating: '4.2', ratingCount: '312', pos: 72, neu: 15, neg: 13,
    img: 'images/baleno.png',
    thumbs: ['images/baleno.png'],
    reviews: [
      { name: 'Amit P', seed: 'Amit', stars: 4, date: 'Feb 2026', text: 'Spacious cabin with premium features at this price range.' },
      { name: 'Neha V', seed: 'Neha', stars: 4, date: 'Jan 2026', text: 'Looks great and drives well. AC could be stronger.' }
    ]
  },
  benz: {
    name: 'C-Class', brand: 'Mercedes', price: '$185,000', tag: 'Luxury Cars',
    rating: '4.5', ratingCount: '519', pos: 78, neu: 14, neg: 8,
    img: 'images/benz.png',
    thumbs: ['images/benz.png'],
    reviews: [
      { name: 'Hans M', seed: 'Hans', stars: 5, date: 'Feb 2026', text: 'Quintessential luxury. The ride quality is unmatched.' },
      { name: 'Clara B', seed: 'Clara', stars: 4, date: 'Jan 2026', text: 'Premium feel throughout. Service costs are high though.' }
    ]
  },
  bmw: {
    name: '6 Series', brand: 'BMW', price: '$220,000', tag: 'Luxury Cars',
    rating: '4.4', ratingCount: '388', pos: 74, neu: 16, neg: 10,
    img: 'images/bmw.png',
    thumbs: ['images/bmw.png'],
    reviews: [
      { name: 'Kevin R', seed: 'Kevin', stars: 5, date: 'Feb 2026', text: 'The driving dynamics are absolutely superb. Pure joy.' },
      { name: 'Anna W', seed: 'Anna', stars: 4, date: 'Jan 2026', text: 'Stunning design and powerful engine. Worth every penny.' }
    ]
  }
};

let currentCarKey = null;
const API_BASE = 'http://localhost:8000';
let allLoadedReviews = [];
let activeReviewTab = 'all';
let selectedSentimentFilter = 'all';
const LOCAL_REVIEW_STORAGE_KEY = 'mumentLocalReviews';
const WISHLIST_STORAGE_KEY = 'mumentWishlist';
const carCategories = {
  audi: 'sports',
  punch: 'suv',
  swift: 'sedan',
  baleno: 'sedan',
  benz: 'sedan',
  bmw: 'sports'
};

function resolveCurrentCarKey() {
  if (currentCarKey && Object.prototype.hasOwnProperty.call(carData, currentCarKey)) {
    return currentCarKey;
  }

  const detailName = document.getElementById('detail-name')?.textContent?.trim();
  if (!detailName) {
    return null;
  }

  const matchedEntry = Object.entries(carData).find(([, car]) => car.name === detailName);
  if (!matchedEntry) {
    return null;
  }

  currentCarKey = matchedEntry[0];
  return currentCarKey;
}

function readLocalReviews() {
  try {
    const parsed = JSON.parse(localStorage.getItem(LOCAL_REVIEW_STORAGE_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error('Error reading local reviews:', error);
    return [];
  }
}

function writeLocalReviews(reviews) {
  localStorage.setItem(LOCAL_REVIEW_STORAGE_KEY, JSON.stringify(reviews));
}

function normalizeReview(review) {
  return {
    id: review.id || `local-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    username: review.username || review.name || 'Guest',
    rating: Number.isFinite(Number(review.rating)) ? Math.max(1, Math.min(5, Number(review.rating))) : 3,
    comment: typeof review.comment === 'string' ? review.comment : '',
    sentiment: typeof review.sentiment === 'string' ? review.sentiment.toLowerCase() : null,
    car_key: review.car_key || null,
    timestamp: review.timestamp || new Date().toISOString(),
    source: review.source || 'local'
  };
}

function mergeReviews(...reviewLists) {
  const merged = new Map();
  reviewLists.flat().forEach((review) => {
    if (!review || typeof review !== 'object') {
      return;
    }

    const normalized = normalizeReview(review);
    if (!normalized.comment.trim()) {
      return;
    }

    const dedupeKey = normalized.id || [normalized.username, normalized.comment, normalized.car_key, normalized.timestamp].join('|');
    merged.set(dedupeKey, normalized);
  });

  return Array.from(merged.values()).sort((left, right) => new Date(right.timestamp) - new Date(left.timestamp));
}

function showHome() {
  document.getElementById('home-page').classList.remove('hidden');
  document.getElementById('detail-page').classList.remove('active');
  document.getElementById('home-page').style.display = '';
  document.getElementById('detail-page').style.display = 'none';
  setActiveNavLink('nav-home');
  window.scrollTo(0, 0);
}

async function showDetail(carKey) {
  if (!Object.prototype.hasOwnProperty.call(carData, carKey)) return;

  currentCarKey = carKey;
  const car = carData[carKey];

  document.getElementById('detail-breadcrumb-name').textContent = car.name;
  document.getElementById('detail-name').textContent = car.name;
  document.getElementById('detail-price').textContent = car.price;
  document.getElementById('detail-tag').textContent = car.tag;
  document.getElementById('d-pos').textContent = car.pos + '%';
  document.getElementById('d-neu').textContent = car.neu + '%';
  document.getElementById('d-neg').textContent = car.neg + '%';
  document.getElementById('d-bar-green').style.flex = car.pos;
  document.getElementById('d-bar-yellow').style.flex = car.neu;
  document.getElementById('d-bar-red').style.flex = car.neg;

  document.getElementById('gallery-main-img').src = car.img;
  const thumbsEl = document.getElementById('gallery-thumbs');
  thumbsEl.innerHTML = car.thumbs.map((t, i) => {
    const safeT = escapeHTML(t);
    return `
      <div class="thumb ${i === 0 ? 'active' : ''}" data-src="${safeT}">
        <img src="${safeT}" alt="">
      </div>`;
  }).join('');

  thumbsEl.querySelectorAll('.thumb').forEach((el) => {
    el.addEventListener('click', function() {
      setThumb(this, this.dataset.src);
    });
  });

  await loadCarReviews(carKey);

  document.getElementById('home-page').style.display = 'none';
  document.getElementById('detail-page').style.display = 'flex';
  document.getElementById('detail-page').classList.add('active');
  setActiveNavLink('nav-reviews');
  window.scrollTo(0, 0);
}

function setActiveNavLink(activeId) {
  document.querySelectorAll('.nav-links a').forEach((link) => {
    link.classList.toggle('active', link.id === activeId);
  });
}

function readWishlist() {
  try {
    const parsed = JSON.parse(localStorage.getItem(WISHLIST_STORAGE_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error('Error reading wishlist:', error);
    return [];
  }
}

function writeWishlist(wishlist) {
  localStorage.setItem(WISHLIST_STORAGE_KEY, JSON.stringify(wishlist));
}

function toggleWishlist(carKey) {
  const currentWishlist = new Set(readWishlist());
  if (currentWishlist.has(carKey)) {
    currentWishlist.delete(carKey);
  } else {
    currentWishlist.add(carKey);
  }

  const updatedWishlist = Array.from(currentWishlist);
  writeWishlist(updatedWishlist);
  syncWishlistButtons(updatedWishlist);
  return updatedWishlist;
}

function syncWishlistButtons(wishlist = readWishlist()) {
  const wishlistSet = new Set(wishlist);
  document.querySelectorAll('.car-card').forEach((card) => {
    const carKey = getCardKey(card);
    const button = card.querySelector('.wishlist-btn');
    if (!button || !carKey) {
      return;
    }

    button.classList.toggle('active', wishlistSet.has(carKey));
  });

  const detailWishlistButton = document.querySelector('.detail-btn.wishlist');
  if (detailWishlistButton && currentCarKey) {
    const isActive = wishlistSet.has(currentCarKey);
    detailWishlistButton.classList.toggle('active', isActive);
    detailWishlistButton.textContent = isActive ? '❤️ In Wishlist' : '❤️ Add to WishList';
  }
}

function getDefaultReviews(carKey) {
  const car = carData[carKey];

  if (!car || !Array.isArray(car.reviews)) {
    return [];
  }

  return car.reviews.map((review) => ({
    id: `default-${carKey}-${review.name}-${review.date}`,
    username: review.name,
    rating: review.stars,
    comment: review.text,
    timestamp: review.date,
    car_key: carKey,
    source: 'default'
  }));
}

function renderReviewCard(review) {
  const safeRating = Number.isFinite(Number(review.rating))
    ? Math.max(1, Math.min(5, Number(review.rating)))
    : 3;
  const username = review.username || review.name || 'Guest';
  const reviewDate = review.timestamp
    ? new Date(review.timestamp).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
    : 'Just now';

  return `
    <div class="review-card">
      <div class="review-card-top">
        <div class="reviewer-avatar">
          <img src="https://api.dicebear.com/7.x/adventurer/svg?seed=${escapeHTML(username)}" alt="">
        </div>
        <div>
          <div class="reviewer-name">${escapeHTML(username)} <span class="verified-badge">♥ Verified</span></div>
          <div class="review-stars">${'★'.repeat(safeRating)}${'<span class="empty-star">★</span>'.repeat(5 - safeRating)}</div>
        </div>
        <span class="review-date">${escapeHTML(reviewDate)}</span>
      </div>
      <div class="review-text">${escapeHTML(review.comment)}</div>
    </div>
  `;
}

async function loadCarReviews(carKey) {
  const reviewsEl = document.getElementById('reviews-list');
  if (!reviewsEl) {
    return;
  }
  const defaultReviews = getDefaultReviews(carKey);
  const localReviews = readLocalReviews().filter((review) => review.car_key === carKey);

  try {
    const response = await fetch(`${API_BASE}/reviews/?page=1&limit=20&car_key=${encodeURIComponent(carKey)}`);
    const payload = await response.json();

    if (!response.ok) {
      allLoadedReviews = mergeReviews(localReviews, defaultReviews);
      renderFilteredReviews();
      return;
    }

    allLoadedReviews = mergeReviews(payload.data || [], localReviews, defaultReviews);
    renderFilteredReviews();
  } catch (error) {
    console.error('Error loading reviews:', error);
    allLoadedReviews = mergeReviews(localReviews, defaultReviews);
    renderFilteredReviews();
  }
}

function renderFilteredReviews() {
  const reviewsEl = document.getElementById('reviews-list');
  if (!reviewsEl) {
    return;
  }

  const filtered = allLoadedReviews.filter((review) => {
    if (activeReviewTab === 'all') {
      return true;
    }
    return (review.sentiment || 'neutral') === activeReviewTab;
  });

  if (!filtered.length) {
    reviewsEl.innerHTML = '<div class="review-card"><div class="review-text">No reviews in this sentiment category yet.</div></div>';
    return;
  }

  reviewsEl.innerHTML = filtered.map(renderReviewCard).join('');
}

function setThumb(el, src) {
  document.querySelectorAll('.thumb').forEach((thumb) => thumb.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('gallery-main-img').src = src;
}

function openReviewModal() {
  resolveCurrentCarKey();
  document.getElementById('review-modal').style.display = 'flex';
}

function closeReviewModal() {
  document.getElementById('review-modal').style.display = 'none';
}

async function submitReview() {
  const stars = [...document.querySelectorAll('#star-rating span')]
    .filter(span => span.textContent === '★').length;
  const reviewText = document.querySelector('#review-modal textarea')?.value.trim();
  const titleInput = document.querySelector('#review-modal input[type="text"]');
  const reviewTitle = titleInput?.value.trim();
  const username = 'Guest';
  const activeCarKey = resolveCurrentCarKey();

  if (!activeCarKey) {
    alert('Open a car detail page before submitting a review.');
    return;
  }

  if (!stars || !reviewText) {
    alert('Please add a rating and review before submitting.');
    return;
  }

  try {
    let analysis = { sentiment: 'neutral', confidence: 0 };
    const analysisResponse = await fetch(`${API_BASE}/reviews/analyse`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        review_text: reviewText
      })
    });

    analysis = await analysisResponse.json().catch(() => ({ sentiment: 'neutral', confidence: 0 }));

    if (!analysisResponse.ok) {
      analysis.sentiment = 'neutral';
      analysis.confidence = 0;
    }

    let savedReview = null;

    const submitResponse = await fetch(`${API_BASE}/reviews/submit`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        username,
        rating: stars,
        comment: reviewTitle ? `${reviewTitle}: ${reviewText}` : reviewText,
        sentiment: analysis.sentiment || 'neutral',
        car_key: activeCarKey
      })

    });

    const submitData = await submitResponse.json().catch(() => ({}));

    if (submitResponse.ok && submitData.review) {
      savedReview = normalizeReview(submitData.review);
    } else {
      savedReview = normalizeReview({
        username,
        rating: stars,
        comment: reviewTitle ? `${reviewTitle}: ${reviewText}` : reviewText,
        sentiment: analysis.sentiment || 'neutral',
        car_key: activeCarKey,
        timestamp: new Date().toISOString(),
        source: 'local-fallback'
      });
    }

    const existingLocalReviews = readLocalReviews();
    const updatedLocalReviews = mergeReviews(existingLocalReviews, [savedReview]);
    writeLocalReviews(updatedLocalReviews);

    const submissionMessage = submitResponse.ok
      ? (submitData.message || 'Review submitted successfully.')
      : 'Review saved locally. Start the backend to persist it server-side.';
    const sentimentText = String(analysis.sentiment || 'neutral');
    const normalizedConfidence = Number(analysis.confidence);
    const confidenceText = Number.isFinite(normalizedConfidence)
      ? ` (${(normalizedConfidence * 100).toFixed(1)}% confidence)`
      : '';

    alert(`${submissionMessage}\nSentiment: ${sentimentText}${confidenceText}`);

    if (titleInput) {
      titleInput.value = '';
    }
    const reviewTextarea = document.querySelector('#review-modal textarea');
    if (reviewTextarea) {
      reviewTextarea.value = '';
    }
    document.querySelectorAll('#star-rating span').forEach((span) => {
      span.textContent = '☆';
      span.style.color = '#ccc';
    });
    activeReviewTab = 'all';
    document.querySelectorAll('.review-tab').forEach((tab) => {
      tab.classList.toggle('active', tab.textContent.trim().toLowerCase() === 'all');
    });
    allLoadedReviews = mergeReviews([savedReview], allLoadedReviews);
    renderFilteredReviews();
    if (activeCarKey) {
      await loadCarReviews(activeCarKey);
    }
    localStorage.setItem('reviewsLastUpdated', String(Date.now()));
    if (document.getElementById('reviews-tbody')) {
      loadDashboard();
    }
    closeReviewModal();
  } catch (error) {
    console.error('Error submitting review:', error);
    alert('An error occurred while submitting your review.');
  }
}

function getCardKey(card) {
  const dataKey = card.dataset.carKey;
  if (dataKey) {
    return dataKey;
  }

  const onclickText = card.getAttribute('onclick') || '';
  const keyMatch = onclickText.match(/showDetail\('([^']+)'\)/);
  return keyMatch ? keyMatch[1] : null;
}

function parseCurrencyValue(rawValue, fallbackValue) {
  const numericValue = Number(String(rawValue || '').replace(/[^\d.]/g, ''));
  return Number.isFinite(numericValue) && numericValue > 0 ? numericValue : fallbackValue;
}

function getCarPriceNumber(carKey) {
  const rawPrice = carData[carKey]?.price || '0';
  return parseCurrencyValue(rawPrice, 0);
}

function getBrandAlias(brand) {
  if (brand === 'benz') {
    return 'mercedes';
  }

  return brand;
}

function updateRatingFilterLabel() {
  const ratingSlider = document.getElementById('rating-range-slider');
  const ratingLabel = document.getElementById('rating-filter-label');
  if (!ratingSlider || !ratingLabel) {
    return;
  }

  const minRating = Math.max(0, Math.min(5, Number(ratingSlider.value) / 20));
  ratingLabel.textContent = `⭐ ${minRating.toFixed(1)}+`;
}

function updateRangeSliderFill(slider) {
  if (!slider) {
    return;
  }

  const min = Number(slider.min || 0);
  const max = Number(slider.max || 100);
  const value = Number(slider.value || min);
  const denominator = max - min;
  const percentage = denominator > 0 ? ((value - min) / denominator) * 100 : 0;
  slider.style.setProperty('--slider-fill', `${Math.max(0, Math.min(100, percentage))}%`);
}

function updatePriceSliderFromInput() {
  const maxInput = document.getElementById('price-max-input');
  const priceSlider = document.getElementById('price-range-slider');
  if (!maxInput || !priceSlider) {
    return;
  }

  const maxPrice = parseCurrencyValue(maxInput.value, 500000);
  const sliderValue = Math.max(0, Math.min(100, Math.round((maxPrice / 500000) * 100)));
  priceSlider.value = String(sliderValue);
  updateRangeSliderFill(priceSlider);
}

function applyHomeFilters() {
  const selectedCategory = document.querySelector('input[name="cat"]:checked')?.value || 'all';
  const selectedBrands = Array.from(document.querySelectorAll('[data-brand] input[type="checkbox"]:checked'))
    .map((input) => input.value.toLowerCase());
  const ratingSlider = document.getElementById('rating-range-slider');
  const minRating = ratingSlider ? (Number(ratingSlider.value) / 20) : 0;
  const minPrice = parseCurrencyValue(document.getElementById('price-min-input')?.value, 50000);
  const maxPrice = parseCurrencyValue(document.getElementById('price-max-input')?.value, 500000);
  const searchTerm = (document.getElementById('nav-search-input')?.value || '').trim().toLowerCase();
  let visibleCount = 0;

  document.querySelectorAll('.car-card').forEach((card) => {
    const cardKey = getCardKey(card);
    const meta = cardKey ? carData[cardKey] : null;
    if (!meta || !cardKey) {
      card.style.display = 'none';
      return;
    }

    const cardCategory = carCategories[cardKey] || 'sedan';
    const cardBrand = getBrandAlias((meta.brand || '').toLowerCase());
    const cardRating = Number(meta.rating || 0);
    const cardPositive = Number(meta.pos || 0);
    const cardPrice = getCarPriceNumber(cardKey);
    const cardSearchText = `${meta.name} ${meta.brand} ${meta.tag}`.toLowerCase();

    const categoryOk = selectedCategory === 'all' || selectedCategory === cardCategory;
    const brandOk = selectedBrands.length === 0 || selectedBrands.includes(cardBrand);
    const ratingOk = cardRating >= minRating;
    const priceOk = cardPrice >= minPrice && cardPrice <= maxPrice;
    const searchOk = !searchTerm || cardSearchText.includes(searchTerm);
    const sentimentOk = selectedSentimentFilter === 'all'
      || (selectedSentimentFilter === 'positive' && cardPositive >= 68)
      || (selectedSentimentFilter === 'neutral' && cardPositive >= 40 && cardPositive < 68)
      || (selectedSentimentFilter === 'negative' && cardPositive < 40);

    const isVisible = categoryOk && brandOk && ratingOk && priceOk && searchOk && sentimentOk;
    card.style.display = isVisible ? '' : 'none';
    if (isVisible) {
      visibleCount += 1;
    }
  });

  const filterSummary = document.getElementById('filter-summary');
  if (filterSummary) {
    filterSummary.textContent = visibleCount
      ? `Showing ${visibleCount} car${visibleCount === 1 ? '' : 's'}`
      : 'No cars match the current filters';
  }
}

function setupHomeSidebarFilters() {
  const applyBtn = document.querySelector('.apply-btn');
  const clearBtn = document.querySelector('.sidebar-header a');
  const homePage = document.getElementById('home-page');

  if (!applyBtn || !homePage) {
    return;
  }

  const sentimentBoxes = document.querySelectorAll('.sentiment-box');
  sentimentBoxes.forEach((box) => {
    box.addEventListener('click', () => {
      const sentiment = box.dataset.sentiment || 'all';
      const isActive = box.classList.contains('active');

      sentimentBoxes.forEach((node) => node.classList.remove('active'));
      selectedSentimentFilter = isActive ? 'all' : sentiment;

      if (!isActive) {
        box.classList.add('active');
      }
    });
  });

  const ratingSlider = document.getElementById('rating-range-slider');
  if (ratingSlider) {
    ratingSlider.addEventListener('input', () => {
      updateRatingFilterLabel();
      updateRangeSliderFill(ratingSlider);
      applyHomeFilters();
    });
    updateRatingFilterLabel();
    updateRangeSliderFill(ratingSlider);
  }

  const priceSlider = document.getElementById('price-range-slider');
  const priceMaxInput = document.getElementById('price-max-input');
  if (priceSlider && priceMaxInput) {
    priceSlider.addEventListener('input', () => {
      const maxPrice = Math.round((Number(priceSlider.value) / 100) * 500000);
      priceMaxInput.value = `$${maxPrice.toLocaleString('en-US')}`;
      updateRangeSliderFill(priceSlider);
      applyHomeFilters();
    });

    updateRangeSliderFill(priceSlider);
  }

  const priceInputs = [document.getElementById('price-min-input'), document.getElementById('price-max-input')];
  priceInputs.forEach((input) => {
    if (!input) {
      return;
    }

    input.addEventListener('blur', () => {
      const fallbackValue = input.id === 'price-min-input' ? 50000 : 500000;
      const normalizedValue = parseCurrencyValue(input.value, fallbackValue);
      input.value = `$${normalizedValue.toLocaleString('en-US')}`;
      if (input.id === 'price-max-input') {
        updatePriceSliderFromInput();
      }
      applyHomeFilters();
    });
  });

  document.querySelectorAll('[data-brand] input[type="checkbox"]').forEach((input) => {
    input.addEventListener('change', applyHomeFilters);
  });

  document.querySelectorAll('input[name="cat"]').forEach((input) => {
    input.addEventListener('change', applyHomeFilters);
  });

  const brandMoreBtn = document.getElementById('brand-more-btn');
  if (brandMoreBtn) {
    brandMoreBtn.addEventListener('click', () => {
      const isExpanded = brandMoreBtn.classList.toggle('expanded');
      document.querySelectorAll('.filter-option-extra').forEach((option) => {
        option.classList.toggle('visible', isExpanded);
      });
      brandMoreBtn.textContent = isExpanded ? 'Less Brands' : 'More Brands';
    });
  }

  applyBtn.addEventListener('click', () => {
    applyHomeFilters();
  });

  if (clearBtn) {
    clearBtn.addEventListener('click', (event) => {
      event.preventDefault();
      selectedSentimentFilter = 'all';

      const allRadio = document.querySelector('input[name="cat"]');
      if (allRadio) {
        allRadio.checked = true;
      }

      document.querySelectorAll('[data-brand] input[type="checkbox"]').forEach((input) => {
        input.checked = ['audi', 'bmw'].includes(input.value.toLowerCase());
      });

      const ratingSliderEl = document.getElementById('rating-range-slider');
      if (ratingSliderEl) {
        ratingSliderEl.value = '60';
        updateRangeSliderFill(ratingSliderEl);
      }

      const priceMinInput = document.getElementById('price-min-input');
      const priceMaxInput = document.getElementById('price-max-input');
      const priceSliderEl = document.getElementById('price-range-slider');
      if (priceMinInput) {
        priceMinInput.value = '$50,000';
      }
      if (priceMaxInput) {
        priceMaxInput.value = '$500,000';
      }
      if (priceSliderEl) {
        priceSliderEl.value = '100';
        updateRangeSliderFill(priceSliderEl);
      }

      sentimentBoxes.forEach((box) => box.classList.remove('active'));
      updateRatingFilterLabel();
      applyHomeFilters();
    });
  }

  applyHomeFilters();
}

function setupNavbarActions() {
  const navCars = document.getElementById('nav-cars');
  const navReviews = document.getElementById('nav-reviews');
  const navAbout = document.getElementById('nav-about');
  const searchInput = document.getElementById('nav-search-input');
  const wishlistButton = document.getElementById('nav-wishlist-btn');
  const notificationsButton = document.getElementById('nav-notifications-btn');
  const viewAllButton = document.querySelector('.view-all');
  const detailWishlistButton = document.querySelector('.detail-btn.wishlist');

  navCars?.addEventListener('click', (event) => {
    event.preventDefault();
    showHome();
    document.getElementById('cars-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setActiveNavLink('nav-cars');
  });

  navReviews?.addEventListener('click', async (event) => {
    event.preventDefault();
    const targetCarKey = currentCarKey || 'audi';
    await showDetail(targetCarKey);
    document.querySelector('.reviews-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setActiveNavLink('nav-reviews');
  });

  navAbout?.addEventListener('click', (event) => {
    event.preventDefault();
    showHome();
    document.getElementById('about-panel')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setActiveNavLink('nav-about');
  });

  searchInput?.addEventListener('input', () => {
    if (document.getElementById('home-page')?.style.display === 'none') {
      showHome();
    }
    setActiveNavLink('nav-cars');
    applyHomeFilters();
  });

  wishlistButton?.addEventListener('click', () => {
    const wishlist = readWishlist();
    if (!wishlist.length) {
      alert('Your wishlist is empty. Tap the heart on any car card to save it.');
      return;
    }

    showHome();
    document.querySelectorAll('[data-brand] input[type="checkbox"]').forEach((input) => {
      input.checked = false;
    });
    document.querySelectorAll('.car-card').forEach((card) => {
      const carKey = getCardKey(card);
      card.style.display = wishlist.includes(carKey) ? '' : 'none';
    });

    const filterSummary = document.getElementById('filter-summary');
    if (filterSummary) {
      filterSummary.textContent = `Showing ${wishlist.length} wishlist car${wishlist.length === 1 ? '' : 's'}`;
    }
    setActiveNavLink('nav-cars');
    document.getElementById('cars-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  notificationsButton?.addEventListener('click', () => {
    const latestReview = mergeReviews(readLocalReviews(), allLoadedReviews)[0];
    if (!latestReview) {
      alert('No new notifications yet.');
      return;
    }

    alert(`Latest review from ${latestReview.username}: ${latestReview.comment}`);
  });

  viewAllButton?.addEventListener('click', () => {
    showHome();
    document.querySelector('.sidebar-header a')?.click();
    setActiveNavLink('nav-cars');
    document.getElementById('cars-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  detailWishlistButton?.addEventListener('click', (event) => {
    event.preventDefault();
    if (!currentCarKey) {
      return;
    }

    const wishlist = toggleWishlist(currentCarKey);
    alert(wishlist.includes(currentCarKey) ? 'Added to wishlist.' : 'Removed from wishlist.');
  });

  document.querySelectorAll('.car-card .wishlist-btn').forEach((button) => {
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      const card = button.closest('.car-card');
      const carKey = card ? getCardKey(card) : null;
      if (!carKey) {
        return;
      }

      toggleWishlist(carKey);
    });
  });

  syncWishlistButtons();
}

function setupAdminSidebarActions() {
  const sidebarItems = document.querySelectorAll('.sidebar .nav-item');
  if (!sidebarItems.length || !document.getElementById('reviews-tbody')) {
    return;
  }

  sidebarItems.forEach((item) => {
    item.addEventListener('click', (event) => {
      const isLogout = item.textContent.toLowerCase().includes('log out');
      if (isLogout) {
        event.preventDefault();
        alert('Logout action placeholder. Connect this to authentication when ready.');
        return;
      }

      event.preventDefault();
      document.querySelectorAll('.sidebar .nav-item').forEach((node) => {
        if (!node.textContent.toLowerCase().includes('log out')) {
          node.classList.remove('active');
        }
      });
      item.classList.add('active');

      const label = item.textContent.trim().toLowerCase();
      if (label.includes('analytics')) {
        window.scrollTo({ top: 0, behavior: 'smooth' });
        loadDashboard();
      } else if (label.includes('messages')) {
        alert('No new admin messages yet.');
      } else if (label.includes('customer review')) {
        window.location.href = 'index.html';
      } else if (label.includes('settings')) {
        alert('Settings panel can be connected here.');
      } else if (label.includes('help')) {
        alert('Help Centre is not connected yet.');
      }
    });
  });
}

function renderDashboardCharts(positive, negative, neutral) {
  if (typeof Chart === 'undefined') {
    return;
  }

  const trendsCanvas = document.getElementById('trendsChart');
  const pieCanvas = document.getElementById('pieChart');

  if (!trendsCanvas || !pieCanvas) {
    return;
  }

  const existingTrendsChart = Chart.getChart(trendsCanvas);
  if (existingTrendsChart) {
    existingTrendsChart.destroy();
  }

  const existingPieChart = Chart.getChart(pieCanvas);
  if (existingPieChart) {
    existingPieChart.destroy();
  }

  const ctx = trendsCanvas.getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Positive', 'Negative', 'Neutral'],
      datasets: [{
        data: [positive, negative, neutral],
        backgroundColor: ['#22c55e', '#ef4444', '#6b7280']
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } }
    }
  });

  const pieCtx = pieCanvas.getContext('2d');
  new Chart(pieCtx, {
    type: 'doughnut',
    data: {
      labels: ['Positive', 'Negative', 'Neutral'],
      datasets: [{
        data: [positive, negative, neutral],
        backgroundColor: ['#22c55e', '#ef4444', '#6b7280'],
        borderWidth: 3,
        borderColor: '#ffffff'
      }]
    },
    options: {
      cutout: '38%',
      plugins: { legend: { display: false } },
      responsive: false
    }
  });
}

async function loadDashboard() {
  const statValues = document.querySelectorAll('.stat-card .stat-value');
  const reviewsTableBody = document.getElementById('reviews-tbody');

  if (!statValues.length || !reviewsTableBody) {
    return;
  }

  try {
    const [summaryRes, reviewsRes] = await Promise.all([
      fetch(`${API_BASE}/analytics/summary`),
      fetch(`${API_BASE}/reviews?page=1&limit=200`)
    ]);

    const summaryPayload = await summaryRes.json().catch(() => ({}));
    const reviewsPayload = await reviewsRes.json().catch(() => ({}));

    let total = Number(summaryPayload.total_reviews) || 0;
    let positive = Number(summaryPayload.positive) || 0;
    let negative = Number(summaryPayload.negative) || 0;
    let neutral = Number(summaryPayload.neutral) || 0;

    if (!summaryRes.ok) {
      console.error('Error loading analytics summary:', summaryPayload);
      total = 0;
      positive = 0;
      negative = 0;
      neutral = 0;
    }

    const apiReviews = reviewsRes.ok && Array.isArray(reviewsPayload.data)
      ? reviewsPayload.data.map(normalizeReview)
      : [];

    if (!reviewsRes.ok) {
      console.error('Error loading reviews for dashboard table:', reviewsPayload);
    }

    const positivePct = total ? ((positive / total) * 100).toFixed(1) : '0.0';
    const negativePct = total ? ((negative / total) * 100).toFixed(1) : '0.0';
    const neutralPct = total ? ((neutral / total) * 100).toFixed(1) : '0.0';

    statValues[0].textContent = total;
    statValues[1].innerHTML = `${positive} <span class="stat-badge badge-green">${positivePct}%</span>`;
    statValues[2].innerHTML = `${negative} <span class="stat-badge badge-red">${negativePct}%</span>`;
    statValues[3].innerHTML = `${neutral} <span class="stat-badge badge-gray">${neutralPct}%</span>`;

    const piePercentages = document.querySelectorAll('.pie-pct');
    if (piePercentages.length >= 3) {
      piePercentages[0].textContent = `${positivePct}%`;
      piePercentages[1].textContent = `${negativePct}%`;
      piePercentages[2].textContent = `${neutralPct}%`;
    }

    reviewsTableBody.innerHTML = '';
    apiReviews.slice(0, 5).forEach((review) => {
      const tr = document.createElement('tr');
      const stars = Array.from({ length: 5 }, (_, index) =>
        `<span class="${index < review.rating ? 'star-filled' : 'star-empty'}">★</span>`
      ).join('');

      tr.innerHTML = `
        <td>
          <div class="td-user">
            <div class="user-avatar-sm">
              <img src="https://api.dicebear.com/7.x/adventurer/svg?seed=${escapeHTML(review.username)}" alt="${escapeHTML(review.username)}">
            </div>
            ${escapeHTML(review.username)}
          </div>
        </td>
        <td><div class="star-rating">${stars}</div></td>
        <td style="color:var(--text-mid)">${escapeHTML(review.comment)}</td>
      `;
      reviewsTableBody.appendChild(tr);
    });

    if (!apiReviews.length) {
      const emptyRow = document.createElement('tr');
      emptyRow.innerHTML = '<td colspan="3" style="color:var(--text-mid)">No reviews available from the backend.</td>';
      reviewsTableBody.appendChild(emptyRow);
    }

    renderDashboardCharts(positive, negative, neutral);
  } catch (error) {
    console.error('Error loading dashboard:', error);
  }
}
// Star rating
const starRating = document.getElementById('star-rating');
if (starRating) {
  starRating.addEventListener('click', function(e) {
    const v = parseInt(e.target.dataset.v);
    if (!v) return;
    this.querySelectorAll('span').forEach((s, i) => s.textContent = i < v ? '★' : '☆');
    this.querySelectorAll('span').forEach((s, i) => s.style.color = i < v ? '#f5a623' : '#ccc');
  });
}

// Review tabs
document.querySelectorAll('.review-tab').forEach(tab => {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.review-tab').forEach(t => t.classList.remove('active'));
    this.classList.add('active');

    const tabLabel = this.textContent.trim().toLowerCase();
    if (tabLabel.startsWith('positive')) {
      activeReviewTab = 'positive';
    } else if (tabLabel.startsWith('neutral')) {
      activeReviewTab = 'neutral';
    } else if (tabLabel.startsWith('negative')) {
      activeReviewTab = 'negative';
    } else {
      activeReviewTab = 'all';
    }

    renderFilteredReviews();
  });
});

// Filter options
document.querySelectorAll('.filter-option').forEach(opt => {
  opt.addEventListener('click', function() {
    if (this.querySelector('input[type=radio]')) {
      document.querySelectorAll('.filter-option').forEach(o => {
        if (o.querySelector('input[type=radio]')) o.classList.remove('active');
      });
      this.classList.add('active');
    }
  });
});

// Close modal on backdrop
const reviewModal = document.getElementById('review-modal');
if (reviewModal) {
  reviewModal.addEventListener('click', function(e) {
    if (e.target === this) closeReviewModal();
  });
}

const darkToggle = document.getElementById('dark-toggle');
if (darkToggle) {
  darkToggle.addEventListener('click', function() {
    document.body.classList.toggle('dark');
  });
}

const refreshBtn = document.getElementById('refresh-btn');
if (refreshBtn) {
  refreshBtn.addEventListener('click', function() {
    location.reload();
  });
}

const notifBtn = document.getElementById('notif-btn');
if (notifBtn) {
  notifBtn.addEventListener('click', function() {
    alert('No new notifications yet.');
  });
}

const webBtn = document.getElementById('web-btn');
if (webBtn) {
  webBtn.addEventListener('click', function() {
    window.open('https://example.com', '_blank', 'noopener,noreferrer');
  });
}

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', function() {
    if (this.style.color === 'rgb(239, 68, 68)') return;
    document.querySelectorAll('.nav-item').forEach(navItem => navItem.classList.remove('active'));
    this.classList.add('active');
  });
});

window.addEventListener('focus', () => {
  if (document.getElementById('reviews-tbody')) {
    loadDashboard();
  }
});

window.addEventListener('storage', (event) => {
  if (event.key === 'reviewsLastUpdated' && document.getElementById('reviews-tbody')) {
    loadDashboard();
  }
});

setupHomeSidebarFilters();
setupNavbarActions();
setupAdminSidebarActions();
loadDashboard();
