// ============================================================
// Notetaker frontend — editorial UI.
// Element IDs are preserved from the old version where practical
// so backend contract doesn't need to change.
// ============================================================

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const uploadForm = document.getElementById("upload-form");
const submitBtn = document.getElementById("submit-btn");
const titleInput = document.getElementById("title-input");
const languageInput = document.getElementById("language-input");

const uploadSection = document.getElementById("upload-section");
const progressSection = document.getElementById("progress-section");
const progressBar = document.getElementById("progress-bar");
const progressStep = document.getElementById("progress-step");
const progressTypewriter = document.getElementById("progress-typewriter");
const progressSubstage = document.getElementById("progress-substage");
const resultSection = document.getElementById("result-section");
const downloadBtn = document.getElementById("download-btn");
const viewBtn = document.getElementById("view-btn");
const newBtn = document.getElementById("new-btn");
const errorSection = document.getElementById("error-section");
const errorMessage = document.getElementById("error-message");
const retryBtn = document.getElementById("retry-btn");

const exampleBtn = document.getElementById("example-btn");
const exampleModal = document.getElementById("example-modal");
const closeModal = document.getElementById("close-modal");
const exampleOutput = document.getElementById("example-output");

const themeToggle = document.getElementById("theme-toggle");

let selectedFile = null;
let currentJobId = null;
let pollInterval = null;

// ========== Theme toggle ==========

try {
    const saved = localStorage.getItem("notetaker-theme");
    if (saved === "dark" || saved === "light") {
        document.documentElement.dataset.theme = saved;
    }
} catch {}

themeToggle.addEventListener("click", () => {
    const root = document.documentElement;
    const current = root.dataset.theme
        || (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("notetaker-theme", next); } catch {}
});

// ========== Dropzone ==========

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        selectFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        selectFile(fileInput.files[0]);
    }
});

const MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024; // 2 GB

function formatBytes(bytes) {
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
}

function selectFile(file) {
    if (file.type && !file.type.startsWith("video/")) {
        showError("Please select a video file.");
        return;
    }
    if (file.size > MAX_FILE_SIZE) {
        showError("File too large (max 2 GB).");
        return;
    }
    selectedFile = file;
    fileName.textContent = `${file.name} · ${formatBytes(file.size)}`;
    fileName.hidden = false;
    submitBtn.disabled = false;
}

// ========== Form submit ==========

uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("video", selectedFile);
    formData.append("title", titleInput.value);
    formData.append("language", languageInput.value);

    uploadSection.hidden = true;
    progressSection.hidden = false;
    resultSection.hidden = true;
    errorSection.hidden = true;
    progressBar.style.width = "0%";
    progressStep.textContent = "Uploading";
    progressTypewriter.textContent = " ";
    progressSubstage.textContent = "";

    try {
        const res = await fetch("/api/convert", {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => null);
            throw new Error(friendlyError(res.status, err?.detail));
        }

        const data = await res.json();
        currentJobId = data.job_id;
        startPolling();
    } catch (err) {
        showError(err.message);
    }
});

// ========== Polling + typewriter reveal ==========

let typingJob = null;
let lastShownSegment = "";

function startPolling() {
    pollInterval = setInterval(pollStatus, 2000);
}

async function pollStatus() {
    if (!currentJobId) return;

    try {
        const res = await fetch(`/api/status/${currentJobId}`);
        if (!res.ok) throw new Error("Failed to fetch status");
        const data = await res.json();

        if (data.total > 0) {
            const pct = Math.round((data.progress / data.total) * 100);
            progressBar.style.width = pct + "%";
        }

        progressStep.textContent = data.stage || "Processing";
        progressSubstage.textContent = data.substage || "";

        // If backend surfaces a recent segment, typewriter-reveal it.
        // Otherwise fall back to the stage detail.
        const toType = data.latest_segment || data.substage || "";
        if (toType && toType !== lastShownSegment) {
            lastShownSegment = toType;
            typewriterReveal(toType);
        }

        if (data.status === "completed") {
            clearInterval(pollInterval);
            stopTyping();
            progressSection.hidden = true;
            resultSection.hidden = false;
        } else if (data.status === "error") {
            clearInterval(pollInterval);
            stopTyping();
            const stage = data.stage ? ` (at: ${data.stage})` : "";
            showError((data.error || "Unknown error") + stage);
        }
    } catch (err) {
        clearInterval(pollInterval);
        stopTyping();
        showError(err.message);
    }
}

function stopTyping() {
    if (typingJob) {
        clearTimeout(typingJob);
        typingJob = null;
    }
}

function typewriterReveal(text) {
    stopTyping();
    // Reveal up to the last ~180 chars of the segment at a realistic reading pace.
    const display = text.length > 260 ? "..." + text.slice(-260) : text;
    progressTypewriter.textContent = "";
    let i = 0;
    const step = () => {
        if (i >= display.length) { typingJob = null; return; }
        progressTypewriter.textContent = display.slice(0, i + 1);
        i += 1;
        typingJob = setTimeout(step, 18);
    };
    step();
}

// ========== Download / view / reset ==========

downloadBtn.addEventListener("click", () => {
    if (currentJobId) {
        window.location.href = `/api/download/${currentJobId}`;
    }
});

viewBtn.addEventListener("click", () => {
    if (currentJobId) {
        window.open(`/api/view/${currentJobId}`, "_blank", "noopener");
    }
});

function resetUI() {
    uploadSection.hidden = false;
    progressSection.hidden = true;
    resultSection.hidden = true;
    errorSection.hidden = true;
    selectedFile = null;
    fileName.textContent = "";
    fileName.hidden = true;
    submitBtn.disabled = true;
    fileInput.value = "";
    currentJobId = null;
    lastShownSegment = "";
    progressTypewriter.textContent = " ";
    progressBar.style.width = "0%";
    stopTyping();
}

newBtn.addEventListener("click", resetUI);
retryBtn.addEventListener("click", resetUI);

// ========== Error ==========

const ERROR_MESSAGES = {
    413: "Video is too large (max 2 GB). Try a shorter or lower-resolution video.",
    429: "Too many requests. Please wait a few minutes and try again.",
    503: "The server is busy. Please try again later.",
    400: "Invalid file. Please upload a supported video format (MP4, MKV, MOV, WebM, AVI).",
};

function friendlyError(status, fallback) {
    return ERROR_MESSAGES[status] || fallback || "Something went wrong.";
}

function showError(msg) {
    progressSection.hidden = true;
    resultSection.hidden = true;
    errorSection.hidden = false;
    errorMessage.textContent = msg;
}

// ========== Example modal ==========

exampleBtn.addEventListener("click", async () => {
    exampleModal.hidden = false;
    if (!exampleOutput.dataset.loaded) {
        try {
            const res = await fetch("/static/example-output.md");
            const text = await res.text();
            exampleOutput.innerHTML = marked.parse(text);
            exampleOutput.dataset.loaded = "1";
        } catch {
            exampleOutput.textContent = "Failed to load the example.";
        }
    }
});

closeModal.addEventListener("click", () => {
    exampleModal.hidden = true;
});

exampleModal.addEventListener("click", (e) => {
    if (e.target === exampleModal) exampleModal.hidden = true;
});

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !exampleModal.hidden) exampleModal.hidden = true;
});
