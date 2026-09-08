// ============================================================
//  IMPORTS
// ============================================================
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile, toBlobURL } from '@ffmpeg/util';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// ============================================================
//  DOM REFS
// ============================================================
const $ = id => document.getElementById(id);
const fileInput = $('fileInput');
const fileZone = $('fileZone');
const fileName = $('fileName');
const fileSize = $('fileSize');
const jsonInput = $('jsonInput');
const btnLoadJSON = $('btnLoadJSON');
const btnClearSegments = $('btnClearSegments');
const segList = $('segList');
const segCount = $('segCount');
const opt916 = $('opt916');
const optDynamic = $('optDynamic');
const btnProcess = $('btnProcess');
const progressWrap = $('progressWrap');
const progressFill = $('progressFill');
const progressText = $('progressText');
const progressDetail = $('progressDetail');
const resultBox = $('resultBox');
const resultVideo = $('resultVideo');
const btnDownload = $('btnDownload');
const toast = $('toast');

// ============================================================
//  STATE
// ============================================================
let segments = [];
let uploadedFile = null;
let ffmpeg = null;
let isProcessing = false;

// ============================================================
//  TOAST & HELPERS
// ============================================================
let toastTimer;
function showToast(msg) {
    clearTimeout(toastTimer);
    toast.textContent = msg;
    toast.classList.add('show');
    toastTimer = setTimeout(() => toast.classList.remove('show'), 3000);
}
function escHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ============================================================
//  RENDER SEGMENTS
// ============================================================
function renderSegments() {
    if (segments.length === 0) {
        segList.innerHTML = `<div class="empty-state">✨ انتظر تحميل JSON لعرض المقاطع</div>`;
        segCount.textContent = '0';
        return;
    }
    let html = '';
    segments.forEach((s, i) => {
        html += `
            <div class="segment-item">
                <span><span class="idx">#${i+1}</span> <span class="time">${escHtml(s.from)} → ${escHtml(s.to)}</span></span>
                <button class="del" data-id="${s.id}">✕</button>
            </div>
        `;
    });
    segList.innerHTML = html;
    segCount.textContent = segments.length;
    document.querySelectorAll('.del').forEach(btn => {
        btn.addEventListener('click', () => {
            const id = parseInt(btn.dataset.id);
            segments = segments.filter(s => s.id !== id);
            renderSegments();
        });
    });
}

// ============================================================
//  FILE UPLOAD
// ============================================================
fileZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.type.startsWith('video/')) {
        showToast('⚠️ الرجاء اختيار ملف فيديو');
        return;
    }
    if (file.size > 200 * 1024 * 1024) {
        showToast('⚠️ حجم الفيديو يتجاوز 200 ميجابايت');
        return;
    }
    uploadedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = `📦 ${(file.size / 1024 / 1024).toFixed(1)} ميجابايت`;
    showToast('✅ تم رفع الفيديو');
});

// ============================================================
//  JSON LOADER
// ============================================================
function loadSegmentsFromJSON() {
    const raw = jsonInput.value.trim();
    if (!raw) { showToast('⚠️ الصق JSON أولاً'); return; }

    let parsed;
    try {
        parsed = JSON.parse(raw);
    } catch (e) {
        showToast('❌ JSON غير صحيح: ' + e.message);
        return;
    }

    if (!Array.isArray(parsed)) {
        showToast('❌ يجب أن تكون مصفوفة (Array)');
        return;
    }

    let addedCount = 0;
    for (const item of parsed) {
        if (item.from && item.to) {
            segments.push({ id: Date.now() + addedCount, from: item.from.trim(), to: item.to.trim() });
            addedCount++;
        }
    }

    if (addedCount === 0) {
        showToast('❌ لم أجد حقول from و to في JSON');
        return;
    }

    renderSegments();
    showToast(`✅ تم استيراد ${addedCount} مقطع بنجاح!`);
    jsonInput.value = '';
}

btnLoadJSON.addEventListener('click', loadSegmentsFromJSON);
btnClearSegments.addEventListener('click', () => {
    if (segments.length === 0) return;
    if (!confirm('مسح جميع المقاطع؟')) return;
    segments = [];
    renderSegments();
});

// ============================================================
//  MEDIAPIPE ANALYSIS (Dynamic Crop)
// ============================================================
async function analyzeFaceTrack(videoFile, duration) {
    const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
    );
    const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'CPU'
        },
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
        numFaces: 1,
        runningMode: 'VIDEO'
    });

    const video = document.createElement('video');
    video.src = URL.createObjectURL(videoFile);
    video.muted = true;
    await video.play();

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const fps = 2;
    const totalFrames = Math.floor(duration * fps);
    const landmarks = [];

    progressDetail.textContent = '🔍 تحليل الوجه (MediaPipe)...';

    for (let i = 0; i < totalFrames; i++) {
        const time = i / fps;
        if (time > duration) break;
        video.currentTime = time;
        await sleep(60);

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = await createImageDataFromCanvas(canvas);

        const result = faceLandmarker.detectForVideo(imageData, performance.now());
        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
            const lm = result.faceLandmarks[0];
            const nose = lm[1];
            const leftEye = lm[2];
            const rightEye = lm[3];
            const cx = (nose.x + leftEye.x + rightEye.x) / 3;
            const cy = (nose.y + leftEye.y + rightEye.y) / 3;
            landmarks.push({ t: time, x: cx, y: cy });
        } else {
            if (landmarks.length > 0) {
                const last = landmarks[landmarks.length - 1];
                landmarks.push({ t: time, x: last.x, y: last.y });
            } else {
                landmarks.push({ t: time, x: 0.5, y: 0.5 });
            }
        }
        const pct = Math.round((i / totalFrames) * 30) + 20;
        progressFill.style.width = pct + '%';
        progressText.textContent = `🔍 تحليل الوجه: ${Math.round(i/totalFrames*100)}%`;
    }

    URL.revokeObjectURL(video.src);
    faceLandmarker.close();
    return landmarks;
}

function createImageDataFromCanvas(canvas) {
    return new Promise((resolve) => {
        canvas.toBlob((blob) => {
            const img = new Image();
            img.onload = () => {
                const c = document.createElement('canvas');
                c.width = img.width;
                c.height = img.height;
                const ctx = c.getContext('2d');
                ctx.drawImage(img, 0, 0);
                resolve(ctx.getImageData(0, 0, c.width, c.height));
            };
            img.src = URL.createObjectURL(blob);
        });
    });
}

function buildDynamicCropCommand(landmarks, totalDuration, videoWidth, videoHeight) {
    const targetRatio = 9 / 16;
    let cropWidth, cropHeight;
    if (videoWidth / videoHeight > targetRatio) {
        cropHeight = videoHeight;
        cropWidth = videoHeight * targetRatio;
    } else {
        cropWidth = videoWidth;
        cropHeight = videoWidth / targetRatio;
    }

    const keyPoints = [];
    const steps = 5;
    for (let i = 0; i < steps; i++) {
        const t = (i / (steps - 1)) * totalDuration;
        let closest = landmarks[0];
        for (const lm of landmarks) {
            if (Math.abs(lm.t - t) < Math.abs(closest.t - t)) closest = lm;
        }
        keyPoints.push({ t, x: closest.x, y: closest.y });
    }

    let filterParts = [];
    let videoInputs = '', audioInputs = '';

    for (let i = 0; i < keyPoints.length - 1; i++) {
        const start = keyPoints[i].t;
        const end = keyPoints[i + 1].t;
        const midX = (keyPoints[i].x + keyPoints[i + 1].x) / 2;
        const midY = (keyPoints[i].y + keyPoints[i + 1].y) / 2;

        const cropX = Math.max(0, Math.min(videoWidth - cropWidth, (midX * videoWidth) - cropWidth / 2));
        const cropY = Math.max(0, Math.min(videoHeight - cropHeight, (midY * videoHeight) - cropHeight / 2));

        const vLabel = `v${i}`;
        const aLabel = `a${i}`;
        filterParts.push(`[0:v]trim=start=${start}:end=${end},setpts=PTS-STARTPTS,crop=${cropWidth}:${cropHeight}:${cropX}:${cropY}[${vLabel}]`);
        filterParts.push(`[0:a]atrim=start=${start}:end=${end},asetpts=PTS-STARTPTS[${aLabel}]`);
        videoInputs += `[${vLabel}]`;
        audioInputs += `[${aLabel}]`;
    }

    const n = keyPoints.length - 1;
    filterParts.push(`${videoInputs}${audioInputs}concat=n=${n}:v=1:a=1[outv][outa]`);
    filterParts.push(`[outv]scale=1080:1920:flags=lanczos[finalv]`);

    return { filterComplex: filterParts.join('; '), mapV: '[finalv]', mapA: '[outa]' };
}

// ============================================================
//  MAIN PROCESS
// ============================================================
async function processVideo() {
    if (isProcessing) return;
    if (!uploadedFile) { showToast('⚠️ اختر فيديو أولاً'); return; }
    if (segments.length === 0) { showToast('⚠️ حمّل JSON للمقاطع أولاً'); return; }

    isProcessing = true;
    btnProcess.disabled = true;
    btnProcess.textContent = '⏳ جاري...';
    progressWrap.classList.add('visible');
    progressFill.style.width = '0%';
    progressText.textContent = '⏳ تحميل المحركات...';
    resultBox.classList.remove('visible');

    try {
        if (!ffmpeg) {
            ffmpeg = new FFmpeg();
            ffmpeg.on('progress', ({ progress }) => {
                const pct = 50 + Math.round(progress * 45);
                progressFill.style.width = Math.min(pct, 95) + '%';
                progressText.textContent = `🎬 معالجة: ${Math.round(progress * 100)}%`;
            });
            // core من CDN (يُجلب بـ fetch عادي وليس Worker، فلا مشكلة cross-origin هنا)
            // فقط @ffmpeg/ffmpeg نفسها كانت تحتاج استضافة محلية بسبب Worker()
            const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm';
            console.log('⏳ جاري تحميل core من:', baseURL);
            await ffmpeg.load({
                coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
                wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
            });
            console.log('✅ ffmpeg.loaded =', ffmpeg.loaded); // يجب أن تكون true هنا
        }

        progressFill.style.width = '5%';
        progressText.textContent = '📁 تجهيز الملف...';
        const inputData = await fetchFile(uploadedFile);
        await ffmpeg.writeFile('input.mp4', inputData); // FS('writeFile', ...) -> writeFile() في v0.12+

        // ===== 1. دمج المقاطع =====
        progressFill.style.width = '10%';
        progressText.textContent = '✂️ دمج المقاطع...';
        const n = segments.length;
        let mergeFilterParts = [], vIns = '', aIns = '';
        segments.forEach((seg, i) => {
            const vLabel = `v${i}`, aLabel = `a${i}`;
            mergeFilterParts.push(`[0:v]trim=start=${seg.from}:end=${seg.to},setpts=PTS-STARTPTS[${vLabel}]`);
            mergeFilterParts.push(`[0:a]atrim=start=${seg.from}:end=${seg.to},asetpts=PTS-STARTPTS[${aLabel}]`);
            vIns += `[${vLabel}]`;
            aIns += `[${aLabel}]`;
        });
        mergeFilterParts.push(`${vIns}${aIns}concat=n=${n}:v=1:a=1[mergedV][mergedA]`);
        await ffmpeg.exec(['-i', 'input.mp4', '-filter_complex', mergeFilterParts.join('; '),
            '-map', '[mergedV]', '-map', '[mergedA]', '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac', 'merged.mp4'
        ]);

        progressFill.style.width = '20%';
        progressText.textContent = '✅ تم الدمج، جاري تحليل الوجه...';

        // ===== 2. تحليل الوجه (Dynamic Crop) =====
        let finalFilter = '', mapV = '[mergedV]', mapA = '[mergedA]';
        if (optDynamic.checked) {
            const mergedData = await ffmpeg.readFile('merged.mp4'); // FS('readFile', ...) -> readFile()
            const mergedBlob = new Blob([mergedData], { type: 'video/mp4' }); // Uint8Array مباشرة، بدون .buffer
            const video = document.createElement('video');
            video.src = URL.createObjectURL(mergedBlob);
            await new Promise((resolve) => { video.onloadedmetadata = resolve;
                video.load(); });
            const vidWidth = video.videoWidth || 1280;
            const vidHeight = video.videoHeight || 720;
            const duration = video.duration || 10;

            const landmarks = await analyzeFaceTrack(mergedBlob, duration);
            if (landmarks.length > 1) {
                const cropResult = buildDynamicCropCommand(landmarks, duration, vidWidth, vidHeight);
                finalFilter = cropResult.filterComplex;
                mapV = cropResult.mapV;
                mapA = cropResult.mapA;
                progressDetail.textContent = '🧠 تم بناء مسار التتبع';
            } else {
                showToast('⚠️ لم يتم اكتشاف وجه، سيتم التوسيط');
            }
            URL.revokeObjectURL(video.src);
        }

        // ===== 3. فلتر 9:16 الثابت (إذا لم يكن ديناميكياً) =====
        if (!finalFilter) {
            if (opt916.checked) {
                finalFilter = `[mergedV]scale=1080:1920:force_original_aspect_ratio=decrease,crop=1080:1920[finalV]`;
                mapV = '[finalV]';
                mapA = '[mergedA]';
            } else {
                finalFilter = `[mergedV]null[finalV]`;
                mapV = '[finalV]';
                mapA = '[mergedA]';
            }
        }

        progressFill.style.width = '50%';
        progressText.textContent = '🎬 تطبيق القص النهائي...';
        await ffmpeg.exec(['-i', 'merged.mp4', '-filter_complex', finalFilter,
            '-map', mapV, '-map', mapA, '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac', 'output.mp4'
        ]);

        progressFill.style.width = '95%';
        progressText.textContent = '📦 قراءة النتيجة...';

        const data = await ffmpeg.readFile('output.mp4'); // FS('readFile', ...) -> readFile()
        const blob = new Blob([data], { type: 'video/mp4' }); // Uint8Array مباشرة، بدون .buffer
        const url = URL.createObjectURL(blob);

        resultVideo.src = url;
        resultVideo.load();
        resultBox.classList.add('visible');

        btnDownload.onclick = () => {
            const a = document.createElement('a');
            a.href = url;
            a.download = 'final_9_16.mp4';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        };

        progressFill.style.width = '100%';
        progressText.textContent = '✅ تم الانتهاء بنجاح!';
        progressDetail.textContent = '🎉 فيديو جاهز مع تتبع الوجه';
        showToast('🎉 فيديو جاهز!');

        // FS('unlink', ...) -> deleteFile() في v0.12+
        await ffmpeg.deleteFile('input.mp4');
        await ffmpeg.deleteFile('merged.mp4');
        await ffmpeg.deleteFile('output.mp4');

    } catch (err) {
        console.error('تفاصيل الخطأ الكاملة:', err);
        // نتعامل مع كل أنواع الأخطاء الممكنة (Error, string, Event, undefined...)
        const errMsg = (err && err.message) ? err.message
                      : (typeof err === 'string') ? err
                      : (err && err.type) ? `حدث خطأ من نوع: ${err.type}`
                      : JSON.stringify(err) || 'خطأ غير معروف (تحقق من Console)';
        showToast('❌ حدث خطأ: ' + errMsg);
        progressText.textContent = '❌ فشل المعالجة';
        progressDetail.textContent = errMsg;
    } finally {
        isProcessing = false;
        btnProcess.disabled = false;
        btnProcess.textContent = '⚡ قص ودمج وتتبع الوجه';
    }
}

btnProcess.addEventListener('click', processVideo);

// ============================================================
//  INIT
// ============================================================
renderSegments();
console.log('🎬 مصنع تتبع الوجه (JSON) جاهز!');
