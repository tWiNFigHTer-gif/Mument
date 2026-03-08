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

function showHome() {
  document.getElementById('home-page').classList.remove('hidden');
  document.getElementById('detail-page').classList.remove('active');
  document.getElementById('home-page').style.display = '';
  document.getElementById('detail-page').style.display = 'none';
  window.scrollTo(0,0);
}

function showDetail(carKey) {
  if (!Object.prototype.hasOwnProperty.call(carData, carKey)) return;
  const car = carData[carKey];

  // Update info
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

  // Gallery
  document.getElementById('gallery-main-img').src = car.img;
  const thumbsEl = document.getElementById('gallery-thumbs');
  thumbsEl.innerHTML = car.thumbs.map((t, i) => {
    const safeT = escapeHTML(t);
    return `
    <div class="thumb ${i===0?'active':''}" data-src="${safeT}">
      <img src="${safeT}" alt="">
    </div>`;
  }).join('');
  thumbsEl.querySelectorAll('.thumb').forEach(el => {
    el.addEventListener('click', function() {
      setThumb(this, this.dataset.src);
    });
  });

  // Reviews
  const reviewsEl = document.getElementById('reviews-list');
  reviewsEl.innerHTML = car.reviews.map(r => `
    <div class="review-card">
      <div class="review-card-top">
        <div class="reviewer-avatar"><img src="https://api.dicebear.com/7.x/adventurer/svg?seed=${escapeHTML(r.seed)}" alt=""></div>
        <div>
          <div class="reviewer-name">${escapeHTML(r.name)} <span class="verified-badge">♥ Verified</span></div>
          <div class="review-stars">${'★'.repeat(r.stars)}${'<span class="empty-star">★</span>'.repeat(5-r.stars)}</div>
        </div>
        <span class="review-date">${escapeHTML(r.date)}</span>
      </div>
      <div class="review-text">${escapeHTML(r.text)}</div>
    </div>`).join('');

  document.getElementById('home-page').style.display = 'none';
  document.getElementById('detail-page').style.display = 'flex';
  document.getElementById('detail-page').classList.add('active');
  window.scrollTo(0,0);
}

function setThumb(el, src) {
  document.querySelectorAll('.thumb').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('gallery-main-img').src = src;
}

function openReviewModal() {
  document.getElementById('review-modal').style.display = 'flex';
}
function closeReviewModal() {
  document.getElementById('review-modal').style.display = 'none';
}
const API_BASE = 'http://localhost:8000';

async function submitReview() {
  const stars = [...document.querySelectorAll('#star-rating span')]
    .filter(span => span.textContent === '★').length;
  const reviewText = document.querySelector('#review-modal textarea')?.value.trim();
  const titleInput = document.querySelector('#review-modal input[type="text"]');
  const reviewTitle = titleInput?.value.trim();
  const username = 'Guest';

  if (!stars || !reviewText) {
    alert('Please add a rating and review before submitting.');
    return;
  }

  try {
    const analysisResponse = await fetch(`${API_BASE}/reviews/analyse`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        review_text: reviewText
      })
    });

    const analysis = await analysisResponse.json();

    if (!analysisResponse.ok) {
      alert('Failed to analyze review.');
      return;
    }

    const submitResponse = await fetch(`${API_BASE}/reviews/submit`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        username,
        rating: stars,
        comment: reviewTitle ? `${reviewTitle}: ${reviewText}` : reviewText,
        sentiment: analysis.sentiment
      })
    });

    const submitData = await submitResponse.json();

    if (!submitResponse.ok) {
      alert('Failed to submit review.');
      return;
    }

    alert(submitData.message || 'Review submitted successfully.');

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

    closeReviewModal();
  } catch (error) {
    console.error('Error submitting review:', error);
    alert('An error occurred while submitting your review.');
  }
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
      fetch(`${API_BASE}/reviews/?page=1&limit=5`)
    ]);

    const summary = await summaryRes.json();
    const reviewsPayload = await reviewsRes.json();

    const total = summary.total_reviews || 0;
    const positive = summary.positive || 0;
    const negative = summary.negative || 0;
    const neutral = summary.neutral || 0;

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
    (reviewsPayload.data || []).forEach((review) => {
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

loadDashboard();
