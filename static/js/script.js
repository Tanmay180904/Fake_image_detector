/**
 * Fake Image Detection System - Client-side JavaScript
 * Handles image preview, form validation, UI interactions, and slider
 */

document.addEventListener('DOMContentLoaded', function() {
    // Image preview functionality
    const imageInput = document.getElementById('imageInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const uploadForm = document.getElementById('uploadForm');
    const detectBtn = document.getElementById('detectBtn');

    // Handle image file selection
    if (imageInput) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            
            if (file) {
                // Validate file type
                const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
                if (!allowedTypes.includes(file.type)) {
                    alert('Please select a valid image file (PNG, JPG, JPEG, GIF, or BMP)');
                    imageInput.value = '';
                    if (previewContainer) previewContainer.style.display = 'none';
                    return;
                }

                // Validate file size (16MB max)
                const maxSize = 16 * 1024 * 1024; // 16MB in bytes
                if (file.size > maxSize) {
                    alert('File size exceeds 16MB. Please select a smaller image.');
                    imageInput.value = '';
                    if (previewContainer) previewContainer.style.display = 'none';
                    return;
                }

                // Display preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    if (imagePreview && previewContainer) {
                        imagePreview.src = e.target.result;
                        previewContainer.style.display = 'block';
                        
                        // Add fade-in animation
                        previewContainer.style.opacity = '0';
                        setTimeout(() => {
                            previewContainer.style.transition = 'opacity 0.3s ease';
                            previewContainer.style.opacity = '1';
                        }, 10);
                    }
                };
                reader.readAsDataURL(file);
            } else {
                if (previewContainer) previewContainer.style.display = 'none';
            }
        });
    }

    // Form submission handling
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const file = imageInput?.files[0];
            
            if (!file) {
                e.preventDefault();
                alert('Please select an image file to upload.');
                return false;
            }

            // Show loading state
            if (detectBtn) {
                const originalText = detectBtn.innerHTML;
                detectBtn.disabled = true;
                detectBtn.innerHTML = '<span class="loading me-2"></span>Processing...';
                
                // Re-enable button after 5 seconds as fallback (in case of error)
                setTimeout(() => {
                    detectBtn.disabled = false;
                    detectBtn.innerHTML = originalText;
                }, 5000);
            }
        });
    }

    // Smooth scroll to top on page load (for result page)
    if (window.location.pathname.includes('predict') || window.location.pathname.includes('result')) {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Animate progress bar on old result layout (safe to keep)
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        const width = progressBar.getAttribute('aria-valuenow');
        progressBar.style.width = '0%';
        
        setTimeout(() => {
            progressBar.style.width = width + '%';
        }, 300);
    }

    // Print functionality enhancement
    window.addEventListener('beforeprint', function() {
        document.body.classList.add('printing');
    });

    window.addEventListener('afterprint', function() {
        document.body.classList.remove('printing');
    });

    // ---- Before / After slider on result page ----
    const compareContainer = document.querySelector('.img-compare-container');
    if (compareContainer) {
        const rightImg = compareContainer.querySelector('.img-right');
        const handle = compareContainer.querySelector('.img-compare-handle');

        let isDragging = false;

        function updateSlider(clientX) {
            const rect = compareContainer.getBoundingClientRect();
            let offsetX = clientX - rect.left;
            offsetX = Math.max(0, Math.min(rect.width, offsetX)); // clamp 0–width

            const percent = (offsetX / rect.width) * 100;

            // move handle
            handle.style.left = percent + '%';
            // reveal right image from that position
            rightImg.style.clipPath = `inset(0 0 0 ${percent}%)`;
        }

        handle.addEventListener('mousedown', (e) => {
            isDragging = true;
            e.preventDefault();
        });

        window.addEventListener('mousemove', (e) => {
            if (isDragging) {
                updateSlider(e.clientX);
            }
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
        });

        // Touch support
        handle.addEventListener('touchstart', (e) => {
            isDragging = true;
            e.preventDefault();
        }, { passive: false });

        window.addEventListener('touchmove', (e) => {
            if (isDragging && e.touches[0]) {
                updateSlider(e.touches[0].clientX);
            }
        }, { passive: false });

        window.addEventListener('touchend', () => {
            isDragging = false;
        });
    }

    // ---- View toggle (ELA vs Grad-CAM) ----
    const buttons = document.querySelectorAll('.btn-toggle');
    const ela = document.getElementById('ela-container');
    const grad = document.getElementById('gradcam-container');

    if (buttons.length && ela && grad) {
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                buttons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                if (btn.dataset.view === 'ela') {
                    ela.style.display = 'block';
                    grad.style.display = 'none';
                } else {
                    ela.style.display = 'none';
                    grad.style.display = 'block';
                }
            });
        });
    }
});
