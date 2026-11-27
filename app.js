const MODEL_PATH = "./models/model.onnx";
const CONFIG_PATH = "./models/config.json";
const IMG_SIZE = 224;

// Default label TrashNet jika config.json tidak mendefinisikan id2label
let labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];
let session = null;
let modelLoadError = null;
let pendingImage = null;

// Data Distribusi Dataset TrashNet
const DATASET_DISTRIBUTION = [
  { name: "Paper (Kertas)", key: "paper", count: 594, pct: 23.5, color: "bg-emerald-500" },
  { name: "Glass (Kaca)", key: "glass", count: 501, pct: 19.8, color: "bg-teal-500" },
  { name: "Plastic (Plastik)", key: "plastic", count: 482, pct: 19.1, color: "bg-sky-500" },
  { name: "Metal (Logam)", key: "metal", count: 410, pct: 16.2, color: "bg-indigo-500" },
  { name: "Cardboard (Kardus)", key: "cardboard", count: 403, pct: 15.9, color: "bg-amber-500" },
  { name: "Trash (Sampah Lainnya)", key: "trash", count: 137, pct: 5.4, color: "bg-rose-500" },
];

// Elemen DOM
const modelBadge = document.getElementById("model-badge");
const imageInput = document.getElementById("image-input");
const dropZone = document.getElementById("drop-zone");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const imagePreview = document.getElementById("image-preview");
const resetBtn = document.getElementById("reset-btn");
const inferenceAlert = document.getElementById("inference-alert");
const resultSection = document.getElementById("result-section");
const topLabel = document.getElementById("top-label");
const topConfidence = document.getElementById("top-confidence");
const probBars = document.getElementById("prob-bars");

// 0. Render Grafik Distribusi Dataset
function renderDatasetDistribution() {
  const datasetContainer = document.getElementById("dataset-bars");
  if (!datasetContainer) return;

  datasetContainer.innerHTML = DATASET_DISTRIBUTION.map(
    (item) => `
      <div>
        <div class="flex justify-between text-xs sm:text-sm mb-1">
          <span class="font-medium text-slate-800">${item.name}</span>
          <span class="text-slate-600 font-semibold">${item.count} <span class="text-slate-400 font-normal">(${item.pct}%)</span></span>
        </div>
        <div class="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
          <div class="${item.color} h-2 rounded-full" style="width: ${(item.pct / 25) * 100}%"></div>
        </div>
      </div>
    `
  ).join("");
}

// 1. Inisialisasi Model & Konfigurasi Lokal
async function initModel() {
  try {
    if (typeof ort === "undefined") {
      throw new Error("Library ONNX Runtime Web belum termuat (periksa koneksi internet).");
    }

    // Load config.json lokal jika tersedia
    try {
      const configRes = await fetch(CONFIG_PATH);
      if (configRes.ok) {
        const config = await configRes.json();
        if (config.id2label) {
          labels = Array.isArray(config.id2label)
            ? config.id2label
            : Object.keys(config.id2label)
                .sort((a, b) => Number(a) - Number(b))
                .map((k) => config.id2label[k]);
        } else if (config.labels) {
          labels = config.labels;
        }
      }
    } catch (_) {
      // Gunakan label default jika config.json tidak ditemukan
    }

    // Load model.onnx lokal
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });

    modelLoadError = null;
    modelBadge.className =
      "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 shrink-0";
    modelBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-500"></span><span>Model Siap</span>`;

    // Jika pengguna sudah memilih gambar saat model masih loading, langsung jalankan prediksi
    if (pendingImage) {
      await runPrediction(pendingImage);
      pendingImage = null;
    }
  } catch (err) {
    console.error("Gagal memuat model lokal:", err);
    modelLoadError = err.message || "File ./models/model.onnx tidak dapat diakses.";
    modelBadge.className =
      "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-rose-50 text-rose-700 border border-rose-200 shrink-0";
    modelBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-500"></span><span>Model Offline</span>`;

    if (pendingImage) {
      showInferenceAlert(
        `Foto berhasil diunggah, tetapi model ONNX gagal dimuat (${modelLoadError}). Pastikan folder <code>./models/model.onnx</code> tersedia dan dijalankan melalui local server (mis. Live Server / <code>python -m http.server</code>).`
      );
    }
  }
}

function showInferenceAlert(htmlMessage) {
  inferenceAlert.innerHTML = htmlMessage;
  inferenceAlert.classList.remove("hidden");
}

function hideInferenceAlert() {
  inferenceAlert.classList.add("hidden");
  inferenceAlert.innerHTML = "";
}

// 2. Preprocessing Gambar -> Tensor NCHW [1, 3, 224, 224]
function preprocessImage(imgElement) {
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d");

  ctx.drawImage(imgElement, 0, 0, IMG_SIZE, IMG_SIZE);
  const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);

  const area = IMG_SIZE * IMG_SIZE;
  const float32Data = new Float32Array(3 * area);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < area; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    float32Data[i] = (r - mean[0]) / std[0];
    float32Data[i + area] = (g - mean[1]) / std[1];
    float32Data[i + 2 * area] = (b - mean[2]) / std[2];
  }

  return new ort.Tensor("float32", float32Data, [1, 3, IMG_SIZE, IMG_SIZE]);
}

// 3. Fungsi Softmax untuk konversi logits ke probabilitas (0 - 1)
function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = Array.from(logits).map((x) => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExps);
}

// 4. Jalankan Prediksi
async function runPrediction(imgElement) {
  if (!session) {
    if (modelLoadError) {
      showInferenceAlert(
        `Foto berhasil diunggah, tetapi model ONNX tidak aktif (${modelLoadError}). Pastikan file <code>./models/model.onnx</code> ada dan diakses via HTTP server.`
      );
    } else {
      pendingImage = imgElement;
      showInferenceAlert("Model ONNX sedang dimuat, hasil analisis akan muncul otomatis sebentar lagi...");
    }
    return;
  }

  try {
    hideInferenceAlert();
    const inputTensor = preprocessImage(imgElement);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const results = await session.run({ [inputName]: inputTensor });
    const rawOutput = results[outputName].data;
    const probabilities = softmax(rawOutput);

    const predictions = labels
      .map((label, idx) => ({
        label,
        score: probabilities[idx] ?? 0,
      }))
      .sort((a, b) => b.score - a.score);

    renderResults(predictions);
  } catch (err) {
    console.error("Kesalahan saat menjalankan inferensi:", err);
    showInferenceAlert(`Terjadi kesalahan saat menganalisis gambar: ${err.message}`);
  }
}

// 5. Render Hasil ke UI
function renderResults(predictions) {
  const best = predictions[0];
  topLabel.textContent = best.label;
  topConfidence.textContent = `${(best.score * 100).toFixed(1)}%`;

  probBars.innerHTML = predictions
    .map((item, idx) => {
      const pct = (item.score * 100).toFixed(1);
      const barColor = idx === 0 ? "bg-emerald-500" : "bg-slate-300";
      const textColor = idx === 0 ? "font-semibold text-slate-900" : "text-slate-500";

      return `
        <div>
          <div class="flex justify-between text-xs mb-1 ${textColor}">
            <span class="capitalize">${item.label}</span>
            <span>${pct}%</span>
          </div>
          <div class="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
            <div class="${barColor} h-1.5 rounded-full transition-all duration-300" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    })
    .join("");

  resultSection.classList.remove("hidden");
}

// 6. Handler Input File, Reset & Drag-and-Drop
function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showInferenceAlert("Format file tidak didukung. Harap unggah file gambar (JPG, PNG, atau WEBP).");
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.onload = () => runPrediction(imagePreview);
    imagePreview.src = e.target.result;
    imagePreview.classList.remove("hidden");
    uploadPlaceholder.classList.add("hidden");
    resetBtn.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
}

imageInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files.length > 0) {
    handleFile(e.target.files[0]);
    // Reset value input agar bisa memilih file yang sama berulang kali
    e.target.value = "";
  }
});

resetBtn.addEventListener("click", () => {
  imageInput.value = "";
  imagePreview.src = "";
  imagePreview.classList.add("hidden");
  uploadPlaceholder.classList.remove("hidden");
  resetBtn.classList.add("hidden");
  resultSection.classList.add("hidden");
  hideInferenceAlert();
  pendingImage = null;
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("border-emerald-500", "bg-slate-100");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("border-emerald-500", "bg-slate-100");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("border-emerald-500", "bg-slate-100");
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});

// Render distribusi dataset & inisialisasi model ONNX
renderDatasetDistribution();
initModel();