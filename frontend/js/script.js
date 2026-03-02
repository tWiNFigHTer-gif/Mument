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
  const car = carData[carKey];
  if (!car) return;

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
  thumbsEl.innerHTML = car.thumbs.map((t, i) => `
    <div class="thumb ${i===0?'active':''}" onclick="setThumb(this,'${t}')">
      <img src="${t}" alt="">
    </div>`).join('');

  // Reviews
  const reviewsEl = document.getElementById('reviews-list');
  reviewsEl.innerHTML = car.reviews.map(r => `
    <div class="review-card">
      <div class="review-card-top">
        <div class="reviewer-avatar"><img src="https://api.dicebear.com/7.x/adventurer/svg?seed=${r.seed}" alt=""></div>
        <div>
          <div class="reviewer-name">${r.name} <span class="verified-badge">♥ Verified</span></div>
          <div class="review-stars">${'★'.repeat(r.stars)}${'<span class="empty-star">★</span>'.repeat(5-r.stars)}</div>
        </div>
        <span class="review-date">${r.date}</span>
      </div>
      <div class="review-text">${r.text}</div>
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
function submitReview() {
  alert('Review submitted successfully! Thank you.');
  closeReviewModal();
}

// Star rating
document.getElementById('star-rating').addEventListener('click', function(e) {
  const v = parseInt(e.target.dataset.v);
  if (!v) return;
  this.querySelectorAll('span').forEach((s, i) => s.textContent = i < v ? '★' : '☆');
  this.querySelectorAll('span').forEach((s, i) => s.style.color = i < v ? '#f5a623' : '#ccc');
});

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
document.getElementById('review-modal').addEventListener('click', function(e) {
  if (e.target === this) closeReviewModal();
});
