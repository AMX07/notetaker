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

function selectFile(file) {
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
            throw new Error(`Upload failed: ${res.statusText}`);
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
            showError(data.error || "Unknown error");
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

function showError(msg) {
    progressSection.hidden = true;
    resultSection.hidden = true;
    errorSection.hidden = false;
    errorMessage.textContent = msg;
}
