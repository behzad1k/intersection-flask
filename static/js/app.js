// Vehicle Counter Web Application
// Main JavaScript
// Get the URL prefix from the template
const URL_PREFIX = window.URL_PREFIX || '';
// Global state
let sessionId = null;
let socket = null;
let videoInfo = null;
let zones = [];
let countingLines = [];
let currentMode = 'zone';
let currentPoints = [];
let canvas = null;
let ctx = null;
let img = null;

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeUpload();
    initializeConfig();
    initializeZoneCanvas();
    initializeWebSocket();
});

// ============ UPLOAD SECTION ============

function initializeUpload() {
    const uploadArea = document.getElementById('upload-area');
    const videoInput = document.getElementById('video-input');
    const browseBtn = document.getElementById('browse-btn');

    // Click to browse
    browseBtn.addEventListener('click', () => videoInput.click());
    uploadArea.addEventListener('click', () => videoInput.click());

    // File selection
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    updateStatus('UPLOADING');

    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch(apiUrl('/upload'), {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const data = await response.json();
        sessionId = data.session_id;
        videoInfo = data;

        // Update UI
        document.getElementById('video-resolution').textContent = `${data.width}x${data.height}`;
        document.getElementById('video-duration').textContent = `${data.duration}s`;
        document.getElementById('video-fps').textContent = data.fps;
        document.getElementById('video-frames').textContent = data.total_frames;

        document.getElementById('upload-area').style.display = 'none';
        document.getElementById('video-info').style.display = 'block';
        document.getElementById('config-panel').style.display = 'block';

        updateStatus('READY');
    } catch (error) {
        console.error('Upload error:', error);
        alert('Failed to upload video. Please try again.');
        updateStatus('ERROR');
    }
}

// ============ CONFIGURATION SECTION ============

function initializeConfig() {
    // Slider updates
    const confSlider = document.getElementById('conf-slider');
    const minDetSlider = document.getElementById('min-det-slider');
    const detWinSlider = document.getElementById('det-win-slider');

    confSlider.addEventListener('input', (e) => {
        document.getElementById('conf-value').textContent = e.target.value;
    });

    minDetSlider.addEventListener('input', (e) => {
        document.getElementById('min-det-value').textContent = e.target.value;
    });

    detWinSlider.addEventListener('input', (e) => {
        document.getElementById('det-win-value').textContent = e.target.value;
    });

    // Next to zones
    document.getElementById('next-to-zones').addEventListener('click', () => {
        loadZoneConfiguration();
    });
}

async function loadZoneConfiguration() {
    updateStatus('LOADING');

    try {
        console.log()
        // Load first frame
        const frameUrl = `${URL_PREFIX}/frame/${sessionId}`;
        img = new Image();
        img.src = frameUrl;

        img.onload = () => {
            initializeCanvas();
            showSection('zone-section');
            updateStatus('CONFIGURING');
        };

        img.onerror = () => {
            alert('Failed to load video frame');
            updateStatus('ERROR');
        };

    } catch (error) {
        console.error('Zone config error:', error);
        alert('Failed to load zone configuration');
        updateStatus('ERROR');
    }
}

// ============ ZONE CANVAS ============

function initializeZoneCanvas() {
    canvas = document.getElementById('zone-canvas');
    ctx = canvas.getContext('2d');

    // Mode switching
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            currentMode = btn.dataset.mode;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update instructions
            if (currentMode === 'zone') {
                document.getElementById('zone-mode-instructions').style.display = 'block';
                document.getElementById('line-mode-instructions').style.display = 'none';
            } else {
                document.getElementById('zone-mode-instructions').style.display = 'none';
                document.getElementById('line-mode-instructions').style.display = 'block';
            }

            currentPoints = [];
            redrawCanvas();
        });
    });

    // Canvas click
    canvas.addEventListener('click', handleCanvasClick);

    // Keyboard
    document.addEventListener('keydown', handleKeyPress);

    // Clear and reset buttons
    document.getElementById('clear-current').addEventListener('click', () => {
        currentPoints = [];
        redrawCanvas();
    });

    document.getElementById('reset-all').addEventListener('click', () => {
        if (confirm('Reset all zones and lines?')) {
            zones = [];
            countingLines = [];
            currentPoints = [];
            redrawCanvas();
            updateZoneCounts();
        }
    });

    // Start processing
    document.getElementById('start-processing').addEventListener('click', startProcessing);
}

function initializeCanvas() {
    const container = canvas.parentElement;

    // Use offsetWidth/offsetHeight as they're more reliable than clientWidth
    // Also add a small delay to ensure layout is complete
    setTimeout(() => {
        const maxWidth = (container.offsetWidth || container.clientWidth || 1220) - 40;
        const maxHeight = window.innerHeight * 0.85;

        console.log('Container dimensions:', {
            offsetWidth: container.offsetWidth,
            clientWidth: container.clientWidth,
            maxWidth: maxWidth
        });

        // Calculate base scale
        const scaleByWidth = maxWidth / img.width;
        const scaleByHeight = maxHeight / img.height;

        // Use the larger scale to fill more space (but cap at reasonable max)
        let scale = Math.max(scaleByWidth, scaleByHeight);

        // For small images, scale up more aggressively
        if (img.width < 800 || img.height < 600) {
            scale = Math.min(scaleByWidth, scaleByHeight) * 1.5; // 1.5x for small images
        }

        // Cap at reasonable maximum to prevent huge canvases
        const maxScale = 3.0; // Allow up to 3x enlargement
        scale = 1.3;
//        scale = Math.min(scale, maxScale);

        // Set canvas size to scaled dimensions
        canvas.width = Math.floor(img.width * scale);
        canvas.height = Math.floor(img.height * scale);

        console.log('Canvas initialized:', {
            imgSize: `${img.width}x${img.height}`,
            canvasSize: `${canvas.width}x${canvas.height}`,
            scale: scale.toFixed(2),
            containerSize: `${maxWidth}x${maxHeight}`
        });

        redrawCanvas();
    }, 50); // 50ms delay to ensure layout is complete
}

function redrawCanvas() {
    if (!img) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / img.width;
    const scaleY = canvas.height / img.height;

    // Draw existing zones
    zones.forEach((zone, index) => {
        ctx.fillStyle = 'rgba(0, 255, 136, 0.2)';
        ctx.strokeStyle = '#00ff88';
        ctx.lineWidth = 2;

        ctx.beginPath();
        zone.forEach((point, i) => {
            const x = point[0] * scaleX;
            const y = point[1] * scaleY;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Draw zone number
        const firstPoint = zone[0];
        ctx.fillStyle = '#00ff88';
        ctx.font = '16px JetBrains Mono';
        ctx.fillText(`Zone ${index + 1}`, firstPoint[0] * scaleX + 5, firstPoint[1] * scaleY - 5);
    });

    // Draw existing lines
    countingLines.forEach((line, index) => {
        const p1 = line.points[0];
        const p2 = line.points[1];

        ctx.strokeStyle = '#ff3b3b';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(p1[0] * scaleX, p1[1] * scaleY);
        ctx.lineTo(p2[0] * scaleX, p2[1] * scaleY);
        ctx.stroke();

        // Draw endpoints
        ctx.fillStyle = '#ff3b3b';
        ctx.beginPath();
        ctx.arc(p1[0] * scaleX, p1[1] * scaleY, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(p2[0] * scaleX, p2[1] * scaleY, 5, 0, Math.PI * 2);
        ctx.fill();

        // Draw label
        const midX = ((p1[0] + p2[0]) / 2) * scaleX;
        const midY = ((p1[1] + p2[1]) / 2) * scaleY;
        ctx.fillStyle = '#ff3b3b';
        ctx.font = '14px JetBrains Mono';
        ctx.fillText(line.direction, midX + 10, midY);
    });

    // Draw current points
    if (currentPoints.length > 0) {
        const color = currentMode === 'zone' ? '#00ff88' : '#00d4ff';
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        currentPoints.forEach((point, i) => {
            const x = point[0] * scaleX;
            const y = point[1] * scaleY;

            // Draw point
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();

            // Draw line to previous point
            if (i > 0) {
                const prevPoint = currentPoints[i - 1];
                ctx.beginPath();
                ctx.moveTo(prevPoint[0] * scaleX, prevPoint[1] * scaleY);
                ctx.lineTo(x, y);
                ctx.stroke();
            }
        });
    }
}

function handleCanvasClick(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert to original image coordinates
    // The canvas displays the image scaled, but we need coordinates in original image space
    const scaleX = img.width / canvas.width;
    const scaleY = img.height / canvas.height;
    const origX = Math.round(x * scaleX);
    const origY = Math.round(y * scaleY);

    currentPoints.push([origX, origY]);

    if (currentMode === 'line' && currentPoints.length === 2) {
        // Complete line
        const direction = prompt('Enter direction name (e.g., "North→South"):');
        if (direction) {
            countingLines.push({
                points: [...currentPoints],
                direction: direction
            });
            currentPoints = [];
            updateZoneCounts();
        }
    }

    redrawCanvas();
}

function handleKeyPress(e) {
    if (e.key === 'Enter' && currentMode === 'zone' && currentPoints.length >= 3) {
        // Complete zone
        zones.push([...currentPoints]);
        currentPoints = [];
        updateZoneCounts();
        redrawCanvas();
    } else if (e.key === 'Escape') {
        // Cancel current drawing
        currentPoints = [];
        redrawCanvas();
    }
}

function updateZoneCounts() {
    document.getElementById('zone-count').textContent = zones.length;
    document.getElementById('line-count').textContent = countingLines.length;

    // Update lists
    const zonesList = document.getElementById('zones-list');
    zonesList.innerHTML = '';
    zones.forEach((zone, i) => {
        const item = document.createElement('div');
        item.className = 'list-item';
        item.textContent = `Zone ${i + 1} (${zone.length} points)`;
        zonesList.appendChild(item);
    });

    const linesList = document.getElementById('lines-list');
    linesList.innerHTML = '';
    countingLines.forEach((line, i) => {
        const item = document.createElement('div');
        item.className = 'list-item';
        item.textContent = line.direction;
        linesList.appendChild(item);
    });
}

// ============ PROCESSING ============

async function startProcessing() {
    if (zones.length === 0) {
        alert('Please define at least one detection zone');
        return;
    }

    if (countingLines.length === 0) {
        alert('Please define at least one counting line');
        return;
    }

    updateStatus('SAVING CONFIG');

    // Save configuration
    try {
        const configResponse = await fetch(apiUrl('/configure'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                zones: zones,
                counting_lines: countingLines
            })
        });

        if (!configResponse.ok) throw new Error('Failed to save configuration');

        // Start processing
        const model = document.getElementById('model-select').value;
        const conf = parseFloat(document.getElementById('conf-slider').value);
        const minDet = parseInt(document.getElementById('min-det-slider').value);
        const detWin = parseInt(document.getElementById('det-win-slider').value);
        const drawBoxes = document.getElementById('draw-boxes').checked;

        const processResponse = await fetch(apiUrl('/process'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                model: model,
                conf_threshold: conf,
                min_detections: minDet,
                detection_window: detWin,
                draw_boxes: drawBoxes
            })
        });

        if (!processResponse.ok) throw new Error('Failed to start processing');

        // Switch to processing view
        showSection('processing-section');
        updateStatus('PROCESSING');

        // Initialize direction counters
        const dirCounters = document.getElementById('direction-counters');
        dirCounters.innerHTML = '';
        countingLines.forEach(line => {
            const counter = document.createElement('div');
            counter.className = 'direction-counter';
            counter.innerHTML = `
                <span class="direction-name">${line.direction}</span>
                <span class="direction-value" id="dir-${line.direction.replace(/[^a-zA-Z0-9]/g, '_')}">0</span>
            `;
            dirCounters.appendChild(counter);
        });

    } catch (error) {
        console.error('Processing error:', error);
        alert('Failed to start processing');
        updateStatus('ERROR');
    }
}

function apiUrl(path) {
    return URL_PREFIX + path;
}

// ============ WEBSOCKET ============

function initializeWebSocket() {
    socket = io({
    path: URL_PREFIX ? `${URL_PREFIX}/socket.io` : '/socket.io'
});

    socket.on('connect', () => {
        console.log('WebSocket connected');
    });

    socket.on('processing_update', (data) => {
        if (data.session_id === sessionId) {
            updateProcessingView(data);
        }
    });

    socket.on('processing_progress', (data) => {
        if (data.session_id === sessionId) {
            updateProgress(data);
        }
    });

    socket.on('processing_complete', (data) => {
        if (data.session_id === sessionId) {
            showResults(data.stats);
        }
    });

    socket.on('processing_error', (data) => {
        if (data.session_id === sessionId) {
            alert('Processing error: ' + data.error);
            updateStatus('ERROR');
        }
    });
}

function updateProcessingView(data) {
    // Update preview frame
    const previewFrame = document.getElementById('preview-frame');
    previewFrame.src = 'data:image/jpeg;base64,' + data.frame;

    // Update counts
    document.getElementById('total-count').textContent = data.counts.total;

    // Update direction counts
    Object.keys(data.counts.by_direction).forEach(direction => {
        const elemId = 'dir-' + direction.replace(/[^a-zA-Z0-9]/g, '_');
        const elem = document.getElementById(elemId);
        if (elem) {
            elem.textContent = data.counts.by_direction[direction];
        }
    });

    // Update class breakdown
    const breakdown = document.getElementById('class-breakdown');
    breakdown.innerHTML = '<h4 style="font-family: JetBrains Mono; font-size: 12px; color: #9ca3af; margin-bottom: 12px; letter-spacing: 1px;">BY CLASS</h4>';

    const allClasses = {};
    Object.values(data.counts.by_class).forEach(dirClasses => {
        Object.keys(dirClasses).forEach(cls => {
            allClasses[cls] = (allClasses[cls] || 0) + dirClasses[cls];
        });
    });

    Object.keys(allClasses).sort().forEach(cls => {
        const item = document.createElement('div');
        item.className = 'class-item';
        item.innerHTML = `
            <span>${cls.toUpperCase()}</span>
            <span class="class-value">${allClasses[cls]}</span>
        `;
        breakdown.appendChild(item);
    });
}

function updateProgress(data) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const frameInfo = document.getElementById('frame-info');
    const fpsInfo = document.getElementById('fps-info');

    progressFill.style.width = data.progress + '%';
    progressText.textContent = Math.round(data.progress) + '%';
    frameInfo.textContent = `Frame ${data.frame_number} / ${data.total_frames}`;
    fpsInfo.textContent = Math.round(data.fps) + ' FPS';
}

// ============ RESULTS ============

function showResults(stats) {
    showSection('results-section');
    updateStatus('COMPLETE');

    // Update summary
    document.getElementById('final-total').textContent = stats.total_all;
    document.getElementById('final-frames').textContent = stats.frames_processed;

    // Create charts
    createDirectionChart(stats.directional_totals);
    createClassChart(stats.directional_breakdown);

    // Download button
    document.getElementById('download-btn').addEventListener('click', () => {
        window.location.href = `${URL_PREFIX}/download/${sessionId}`;
    });

    // New analysis button
    document.getElementById('new-analysis-btn').addEventListener('click', () => {
        location.reload();
    });
}

function createDirectionChart(directionalTotals) {
    const ctx = document.getElementById('direction-chart').getContext('2d');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(directionalTotals),
            datasets: [{
                label: 'Vehicles',
                data: Object.values(directionalTotals),
                backgroundColor: 'rgba(0, 212, 255, 0.6)',
                borderColor: '#00d4ff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: { color: '#e5e7eb', font: { family: 'JetBrains Mono' } }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#9ca3af', font: { family: 'JetBrains Mono' } },
                    grid: { color: '#1f2937' }
                },
                x: {
                    ticks: { color: '#9ca3af', font: { family: 'JetBrains Mono' } },
                    grid: { color: '#1f2937' }
                }
            }
        }
    });
}

function createClassChart(directionalBreakdown) {
    const classData = {};

    Object.values(directionalBreakdown).forEach(dirData => {
        Object.keys(dirData).forEach(cls => {
            classData[cls] = (classData[cls] || 0) + dirData[cls];
        });
    });

    const ctx = document.getElementById('class-chart').getContext('2d');

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(classData),
            datasets: [{
                data: Object.values(classData),
                backgroundColor: [
                    '#00d4ff',
                    '#00ff88',
                    '#ffd93b',
                    '#ff3b3b',
                    '#ff00ff',
                    '#ff9900',
                    '#00ffff',
                    '#888888'
                ],
                borderColor: '#151b24',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#e5e7eb', font: { family: 'JetBrains Mono', size: 11 } }
                }
            }
        }
    });
}

// ============ UTILITIES ============

function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(sectionId).classList.add('active');
}

function updateStatus(status) {
    const statusElem = document.getElementById('status');
    statusElem.textContent = status;

    // Update color based on status
    const colors = {
        'READY': '--green',
        'UPLOADING': '--yellow',
        'LOADING': '--yellow',
        'CONFIGURING': '--cyan',
        'SAVING CONFIG': '--yellow',
        'PROCESSING': '--cyan',
        'COMPLETE': '--green',
        'ERROR': '--red'
    };

    const color = colors[status] || '--cyan';
    statusElem.style.borderColor = `var(${color})`;
    statusElem.style.color = `var(${color})`;
}