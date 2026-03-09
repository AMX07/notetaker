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
const progressStage = document.getElementById("progress-stage");
const progressSubstage = document.getElementById("progress-substage");
const resultSection = document.getElementById("result-section");
const downloadBtn = document.getElementById("download-btn");
const newBtn = document.getElementById("new-btn");
const errorSection = document.getElementById("error-section");
const errorMessage = document.getElementById("error-message");
const retryBtn = document.getElementById("retry-btn");

const exampleBtn = document.getElementById("example-btn");
const exampleModal = document.getElementById("example-modal");
const closeModal = document.getElementById("close-modal");
const exampleOutput = document.getElementById("example-output");

let selectedFile = null;
let currentJobId = null;
let pollInterval = null;

// --- Dropzone ---

dropzone.addEventListener("click", () => fileInput.click());

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
const ALLOWED_TYPES = ["video/mp4", "video/x-matroska", "video/quicktime", "video/webm", "video/x-msvideo"];

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
    fileName.textContent = file.name;
    submitBtn.disabled = false;
}

// --- Form Submit ---

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
    progressStage.textContent = "Uploading...";

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

// --- Polling ---

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
        progressStage.textContent = data.stage || "Processing...";
        progressSubstage.textContent = data.substage || "";

        if (data.status === "completed") {
            clearInterval(pollInterval);
            progressSection.hidden = true;
            resultSection.hidden = false;
        } else if (data.status === "error") {
            clearInterval(pollInterval);
            const stage = data.stage ? ` (at: ${data.stage})` : "";
            showError((data.error || "Unknown error") + stage);
        }
    } catch (err) {
        clearInterval(pollInterval);
        showError(err.message);
    }
}

// --- Download ---

downloadBtn.addEventListener("click", () => {
    if (currentJobId) {
        window.location.href = `/api/download/${currentJobId}`;
    }
});

// --- Reset ---

function resetUI() {
    uploadSection.hidden = false;
    progressSection.hidden = true;
    resultSection.hidden = true;
    errorSection.hidden = true;
    selectedFile = null;
    fileName.textContent = "";
    submitBtn.disabled = true;
    fileInput.value = "";
    currentJobId = null;
}

newBtn.addEventListener("click", resetUI);
retryBtn.addEventListener("click", resetUI);

// --- Error ---

const ERROR_MESSAGES = {
    413: "Video is too large (max 2 GB). Try a shorter or lower-resolution video.",
    429: "Too many requests. Please wait a few minutes and try again.",
    503: "Server is busy. Please try again later.",
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

// --- Example Modal ---

exampleBtn.addEventListener("click", async () => {
    exampleModal.hidden = false;
    if (!exampleOutput.dataset.loaded) {
        try {
            const res = await fetch("/static/example-output.md");
            const text = await res.text();
            exampleOutput.textContent = text;
            exampleOutput.dataset.loaded = "1";
        } catch {
            exampleOutput.textContent = "Failed to load example.";
        }
    }
});

closeModal.addEventListener("click", () => {
    exampleModal.hidden = true;
});

exampleModal.addEventListener("click", (e) => {
    if (e.target === exampleModal) exampleModal.hidden = true;
});
